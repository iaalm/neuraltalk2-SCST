
require 'torch'
require 'nn'
require 'nngraph'
-- exotic things
require 'loadcaffe'
-- local imports
local utils = require 'misc.utils'
require 'misc.DataLoader'
require 'misc.LanguageModel'
local net_utils = require 'misc.net_utils'
require 'misc.optim_updates'

-------------------------------------------------------------------------------
-- Input arguments and options
-------------------------------------------------------------------------------
cmd = torch.CmdLine()
cmd:text()
cmd:text('Train an Image Captioning model')
cmd:text()
cmd:text('Options')

-- Data input settings
cmd:option('-input_h5','/s/coco/cocotalk.h5','path to the h5file containing the preprocessed dataset')
cmd:option('-input_val','annotations/captions_val2014.json','path to the json file containing caption for val')
cmd:option('-input_json','/s/coco/cocotalk.json','path to the json file containing additional info and vocab')
cmd:option('-start_from', '../../model_.t7', 'path to a model checkpoint to initialize model weights from. Empty = don\'t')

-- Model settings
cmd:option('-rnn_size',512,'size of the rnn in number of hidden nodes in each layer')
cmd:option('-input_encoding_size',512,'the encoding size of each token in the vocabulary, and the image.')

-- Optimization: General
cmd:option('-max_iters', 200000, 'max number of iterations to run for (-1 = run forever)')
cmd:option('-batch_size',32,'what is the batch size in number of images per batch? (there will be x seq_per_img sentences)')
cmd:option('-grad_clip',0.1,'clip gradients at this value (note should be lower than usual 5 because we normalize grads by both batch and seq_length)')
cmd:option('-seq_per_img',5,'number of captions to sample for each image during training. Done for efficiency since CNN forward pass is expensive. E.g. coco has 5 sents/image')
cmd:option('-finetune_cnn_after', -1, 'After what iteration do we start finetuning the CNN? (-1 = disable; never finetune, 0 = finetune from start)')
-- Optimization: for the Language Model
cmd:option('-optim','adam','what update to use? rmsprop|sgd|sgdmom|adagrad|adam')
cmd:option('-learning_rate',5e-5,'learning rate')
cmd:option('-learning_rate_decay_start', 0, 'at what iteration to start decaying learning rate? (-1 = dont)')
cmd:option('-learning_rate_decay_every', 10000, 'every how many iterations thereafter to drop LR by half?')
cmd:option('-optim_alpha',0.8,'alpha for adagrad/rmsprop/momentum/adam')
cmd:option('-optim_beta',0.999,'beta used for adam')
cmd:option('-optim_epsilon',1e-8,'epsilon that goes into denominator for smoothing')
-- Optimization: for the CNN
cmd:option('-cnn_optim','adam','optimization to use for CNN')
cmd:option('-cnn_optim_alpha',0.8,'alpha for momentum of CNN')
cmd:option('-cnn_optim_beta',0.999,'alpha for momentum of CNN')
cmd:option('-cnn_learning_rate',1e-6,'learning rate for the CNN')
cmd:option('-cnn_weight_decay', 0, 'L2 weight decay just for the CNN')

-- Evaluation/Checkpointing
cmd:option('-val_images_use', 3200, 'how many images to use when periodically evaluating the validation loss? (-1 = all)')
cmd:option('-save_checkpoint_every', 500, 'how often to save a model checkpoint?')
cmd:option('-checkpoint_path', '', 'folder to save checkpoints into (empty = this folder)')
cmd:option('-language_eval', 1, 'Evaluate language as well (1 = yes, 0 = no)? BLEU/CIDEr/METEOR/ROUGE_L? requires coco-caption code from Github.')
cmd:option('-losses_log_every', 25, 'How often do we snapshot losses, for inclusion in the progress dump? (0 = disable)')

-- misc
cmd:option('-backend', 'cudnn', 'nn|cudnn')
cmd:option('-id', '', 'an id identifying this run/job. used in cross-val and appended when writing progress files')
cmd:option('-seed', 123, 'random number generator seed to use')
cmd:option('-gpuid', 0, 'which gpu to use. -1 = use CPU')

cmd:option('-distrub_lable', 0, 'distrub lable')
cmd:option('-beam_size', 1, 'beam search size')

cmd:text()

-------------------------------------------------------------------------------
--   /$$$$$$           /$$   /$$           /$$$$$$$                        /$$    
--  |_  $$_/          |__/  | $$          | $$__  $$                      | $$    
--    | $$   /$$$$$$$  /$$ /$$$$$$        | $$  \ $$  /$$$$$$   /$$$$$$  /$$$$$$  
--    | $$  | $$__  $$| $$|_  $$_/        | $$$$$$$/ |____  $$ /$$__  $$|_  $$_/  
--    | $$  | $$  \ $$| $$  | $$          | $$____/   /$$$$$$$| $$  \__/  | $$    
--    | $$  | $$  | $$| $$  | $$ /$$      | $$       /$$__  $$| $$        | $$ /$$
--   /$$$$$$| $$  | $$| $$  |  $$$$/      | $$      |  $$$$$$$| $$        |  $$$$/
--  |______/|__/  |__/|__/   \___/        |__/       \_______/|__/         \___/                               
-------------------------------------------------------------------------------
local task_hash = torch.random()
print('task hash:', task_hash)
local opt = cmd:parse(arg)
torch.manualSeed(opt.seed)
torch.setdefaulttensortype('torch.FloatTensor') -- for CPU

local checkpoint_path = path.join(opt.checkpoint_path, 'model_' .. opt.id)
if (path.exists(checkpoint_path .. '.json')) then
  print('logfile ' .. checkpoint_path .. '.json exists !')
  os.exit(1)
end

if opt.gpuid >= 0 then
  require 'cutorch'
  require 'cunn'
  if opt.backend == 'cudnn' then require 'cudnn' end
  cutorch.manualSeed(opt.seed)
  cutorch.setDevice(opt.gpuid + 1) -- note +1 because lua is 1-indexed
end
-------------------------------------------------------------------------------
-- Create the Data Loader instance
-------------------------------------------------------------------------------
local loader = DataLoader{h5_file = opt.input_h5, json_file = opt.input_json}
local Zvocab = loader:getVocabSize()
-------------------------------------------------------------------------------
-- Initialize the networks
-------------------------------------------------------------------------------
local protos = {}

-- load protos from file
print('initializing weights from ' .. opt.start_from)
local loaded_checkpoint = torch.load(opt.start_from)
protos = loaded_checkpoint.protos
net_utils.unsanitize_gradients(protos.cnn)
local lm_modules = protos.lm:getModulesList()
for k,v in pairs(lm_modules) do net_utils.unsanitize_gradients(v) end
protos.crit = nn.LanguageModelCriterion() -- not in checkpoints, create manually
protos.expander = nn.FeatExpander(opt.seq_per_img) -- not in checkpoints, create manually

-- ship everything to GPU, maybe
if opt.gpuid >= 0 then
  for k,v in pairs(protos) do v:cuda() end
end

-- flatten and prepare all model parameters to a single vector. 
-- Keep CNN params separate in case we want to try to get fancy with different optims on LM/CNN
local params, grad_params = protos.lm:getParameters()
local cnn_params, cnn_grad_params = protos.cnn:getParameters()
print('total number of parameters in LM: ', params:nElement())
print('total number of parameters in CNN: ', cnn_params:nElement())
assert(params:nElement() == grad_params:nElement())
assert(cnn_params:nElement() == cnn_grad_params:nElement())

-- construct thin module clones that share parameters with the actual
-- modules. These thin module will have no intermediates and will be used
-- for checkpointing to write significantly smaller checkpoint files
local thin_lm = protos.lm:clone()
thin_lm.core:share(protos.lm.core, 'weight', 'bias') -- TODO: we are assuming that LM has specific members! figure out clean way to get rid of, not modular.
thin_lm.lookup_table:share(protos.lm.lookup_table, 'weight', 'bias')
local thin_cnn = protos.cnn:clone('weight', 'bias')
-- sanitize all modules of gradient storage so that we dont save big checkpoints
net_utils.sanitize_gradients(thin_cnn)
local lm_modules = thin_lm:getModulesList()
for k,v in pairs(lm_modules) do net_utils.sanitize_gradients(v) end

-- create clones and ensure parameter sharing. we have to do this 
-- all the way here at the end because calls such as :cuda() and
-- :getParameters() reshuffle memory around.
protos.lm:createClones()

collectgarbage() -- "yeah, sure why not"


-------------------------------------------------------------------------------
--   /$$$$$$$            /$$            /$$$$$$                                         
--  | $$__  $$          |__/           /$$__  $$                                        
--  | $$  \ $$  /$$$$$$  /$$ /$$$$$$$ | $$  \__/  /$$$$$$   /$$$$$$   /$$$$$$$  /$$$$$$ 
--  | $$$$$$$/ /$$__  $$| $$| $$__  $$| $$$$     /$$__  $$ /$$__  $$ /$$_____/ /$$__  $$
--  | $$__  $$| $$$$$$$$| $$| $$  \ $$| $$_/    | $$  \ $$| $$  \__/| $$      | $$$$$$$$
--  | $$  \ $$| $$_____/| $$| $$  | $$| $$      | $$  | $$| $$      | $$      | $$_____/
--  | $$  | $$|  $$$$$$$| $$| $$  | $$| $$      |  $$$$$$/| $$      |  $$$$$$$|  $$$$$$$
--  |__/  |__/ \_______/|__/|__/  |__/|__/       \______/ |__/       \_______/ \_______/            
-------------------------------------------------------------------------------
torch.class('nlp')
local simple_metric, parent = torch.class('nlp.simple_metric')

function simple_metric:__init()
end


function simple_metric:eval(seq, label)
	local function seqLen(seq)
		for i=1, seq:size(1) do
			if seq[i] == 0 or seq[i] == Zvocab+1 then
				return i - 1
			end
		end
		return seq:size(1)
	end

	local function seqMatch(src, label, start, n)
		start = start - 1
		local l = seqLen(src)
		for i=0,l-n do
			flag = true
			for j=1,n do
				if src[i+j] ~= label[start+j] then
					flag = false
					break
				end
			end
			if flag then
				return true
			end
		end
		return false
	end

	seq = seq:t()
	label = label:t()
	local B = seq:size(1)
	local S = seq:size(2)
	local gain = torch.Tensor(B)
	assert(label:size(1) == B * 5 and label:size(2) == S)
	local n = 4
	for b=1,B do
		local match = 0
		local count = seqLen(seq[b]) - n + 1
	    for i=1,count do
	    	for j=1,5 do
	    		if seqMatch(label[b*5-5+j], seq[b], i, n) then
	    			match = match + 1
	    			break
	    		end
	    	end
	    end
	    if count == 0 then
	    	gain[b] = 0
		else
			local length_total = 0
			for j=1,5 do
	    		length_total = length_total + seqLen(label[b*5-5+j])
	    	end
	    	local penalty = math.exp(1 - math.max(1, length_total / seqLen(seq[b]) / 5))
	    	gain[b] = match / count * penalty
		end
	  end
	  return gain
end

local bleu_metric, parent = torch.class('nlp.bleu_metric')
function bleu_metric:__init(n)
  self.n = n
end

function bleu_metric:eval(seq, label)
  local eos = loader:getVocabSize() + 1
	local function seqLen(seq)
		for i=1, seq:size(1) do
			if seq[i] == 0 or seq[i] == eos then
        seq[i] = eos
				return i
			end
		end
		return seq:size(1)
  end
  local function seqMatch(seq, ref)
    local n = seq:size(1)
    local l = seqLen(ref)
    local count = 0
    for i=1,l-n+1 do
      if seq:equal(ref[{{i, i+n-1}}]) then
        count = count + 1
      end
    end
    local vocab = loader:getVocab()
    -- print(count)
    -- print(net_utils.decode_sequence(vocab, seq:reshape(n,1))[1])
    -- print(net_utils.decode_sequence(vocab, ref:reshape(ref:size(1), 1))[1])
    return count
  end
  -- local vocab = loader:getVocab()
  -- print(net_utils.decode_sequence(vocab, seq))
  -- print(net_utils.decode_sequence(vocab, label))
	seq = seq:t()
	label = label:t()
	local B = seq:size(1)
	local S = seq:size(2)
	local gain = torch.Tensor(B)
	assert(label:size(1) == B * 5 and label:size(2) == S)
  for b=1,B do
    local bleu = 1
    local bleus = {}
    for n=1,self.n do
      local l = seqLen(seq[b])
      local guess = 1e-8
      local correct = 1e-12
      for i=1,l-n+1 do
        local self_count = seqMatch(seq[{b, {i,i+n-1}}], seq[b])
        local max_count = 0
        local ref_min_len = 100
        for j=1,5 do
          local ref_len = seqLen(label[b*5+j-5])
          if ref_len < ref_min_len then ref_min_len = ref_len end
          local c = seqMatch(seq[{b, {i,i+n-1}}], label[b*5+j-5])
          if c > max_count then max_count = c end
        end
        assert(self_count >= 1)
        guess = guess + 1 / self_count
        correct = correct + math.min(max_count, self_count) / self_count / self_count * math.exp(math.min(0, 1 - ref_min_len/l))
      end
      bleu = bleu * correct / guess
      table.insert(bleus, math.pow(bleu, 1/n))
    end
    gain[b] = math.pow(bleu, 1/self.n)
  end
  return gain
end


function policy_grad(gain, sample_seq, log_prob)
  local prob = torch.exp(log_prob)
	local B = sample_seq:size(2)
	local S = sample_seq:size(1)
  local V = log_prob:size(3)
  local grad = torch.FloatTensor(S+2, B, V)
  local n = 0
  grad[{{2,1+S},{},{}}] = prob
  grad[{{1},{},{}}] = 0
  grad[{{S+2},{},{}}] = 0
	assert(gain:dim() == 1 and gain:size(1) == B)
  for b=1,B do
    for s=1,S+1 do
      local idx
      if s > S then 
        idx = Zvocab+1
      else
        idx = sample_seq[s][b]
      end
      grad[s+1][b][idx] = grad[s+1][b][idx] - 1
      n = n + 1
      if idx == Zvocab+1 then 
        break
      end
    end
  end
  local gain_mask = gain:repeatTensor(S+2, V, 1):transpose(2,3)
	return grad:cmul(gain_mask):div(n)

end
function policy_grad_old(gain, sample_seq)
	local B = sample_seq:size(2)
	local S = sample_seq:size(1)
  local grad = torch.Tensor(S+2, B, Zvocab + 1):zero():cuda()
	assert(gain:dim() == 1 and gain:size(1) == B)
	-- grad:scatter(2, sample_seq, -gain:repeatTensor(S, 1):t())
  for b=1,B do
    for s=1,S do
      local idx = sample_seq[s][b]
      grad[s+1][b][idx] = - gain[b]
      if idx == Zvocab+1 then 
        break
      end
    end
  end
	return grad:div(B)
end

local metric = nlp.bleu_metric(4)
-------------------------------------------------------------------------------
--   /$$$$$$$$                      /$$       /$$$$$$$$                              
--  | $$_____/                     | $$      | $$_____/                              
--  | $$       /$$    /$$  /$$$$$$ | $$      | $$       /$$   /$$ /$$$$$$$   /$$$$$$$
--  | $$$$$   |  $$  /$$/ |____  $$| $$      | $$$$$   | $$  | $$| $$__  $$ /$$_____/
--  | $$__/    \  $$/$$/   /$$$$$$$| $$      | $$__/   | $$  | $$| $$  \ $$| $$      
--  | $$        \  $$$/   /$$__  $$| $$      | $$      | $$  | $$| $$  | $$| $$      
--  | $$$$$$$$   \  $/   |  $$$$$$$| $$      | $$      |  $$$$$$/| $$  | $$|  $$$$$$$
--  |________/    \_/     \_______/|__/      |__/       \______/ |__/  |__/ \_______/               
-------------------------------------------------------------------------------
local function eval_split(split, evalopt)
  local verbose = utils.getopt(evalopt, 'verbose', true)
  local val_images_use = utils.getopt(evalopt, 'val_images_use', true)

  protos.cnn:evaluate()
  protos.lm:evaluate()
  loader:resetIterator(split) -- rewind iteator back to first datapoint in the split
  local n = 0
  local loss_sum = 0
  local loss_evals = 0
  local predictions = {}
  local vocab = loader:getVocab()
  while true do

    -- fetch a batch of data
    local data = loader:getBatch{batch_size = opt.batch_size, split = split, seq_per_img = opt.seq_per_img}
    data.images = net_utils.prepro(data.images, false, opt.gpuid >= 0) -- preprocess in place, and don't augment
    n = n + data.images:size(1)

    -- forward the model to get loss
    local feats = protos.cnn:forward(data.images)
    local expanded_feats = protos.expander:forward(feats)
    local logprobs = protos.lm:forward{expanded_feats, data.labels}
    local loss = protos.crit:forward(logprobs, data.labels)
    loss_sum = loss_sum + loss
    loss_evals = loss_evals + 1

    -- forward the model to also get generated samples for each image
    local seq = protos.lm:sample(feats, {beam_size=opt.beam_size})
    local sents = net_utils.decode_sequence(vocab, seq)
    for k=1,#sents do
      local entry = {image_id = data.infos[k].id, caption = sents[k]}
      table.insert(predictions, entry)
      if verbose then
        print(string.format('image %s: %s', entry.image_id, entry.caption))
      end
    end

    -- if we wrapped around the split or used up val imgs budget then bail
    local ix0 = data.bounds.it_pos_now
    local ix1 = math.min(data.bounds.it_max, val_images_use)
    if verbose then
      print(string.format('evaluating validation performance... %d/%d (%f)', ix0-1, ix1, loss))
    end

    if loss_evals % 10 == 0 then collectgarbage() end
    if data.bounds.wrapped then break end -- the split ran out of data, lets break out
    if n >= val_images_use then break end -- we've used enough images
  end

  local lang_stats
  if opt.language_eval == 1 then
    lang_stats = net_utils.language_eval(predictions, opt.input_val, task_hash)
  end

  return loss_sum/loss_evals, predictions, lang_stats
end

-------------------------------------------------------------------------------
--   /$$$$$$$$                     /$$                 /$$$$$$$$                              
--  |__  $$__/                    |__/                | $$_____/                              
--     | $$     /$$$$$$   /$$$$$$  /$$ /$$$$$$$       | $$       /$$   /$$ /$$$$$$$   /$$$$$$$
--     | $$    /$$__  $$ |____  $$| $$| $$__  $$      | $$$$$   | $$  | $$| $$__  $$ /$$_____/
--     | $$   | $$  \__/  /$$$$$$$| $$| $$  \ $$      | $$__/   | $$  | $$| $$  \ $$| $$      
--     | $$   | $$       /$$__  $$| $$| $$  | $$      | $$      | $$  | $$| $$  | $$| $$      
--     | $$   | $$      |  $$$$$$$| $$| $$  | $$      | $$      |  $$$$$$/| $$  | $$|  $$$$$$$
--     |__/   |__/       \_______/|__/|__/  |__/      |__/       \______/ |__/  |__/ \_______/ 
-------------------------------------------------------------------------------
local iter = 0
local function lossFun()
  protos.cnn:training()
  protos.lm:training()
  grad_params:zero()
  if opt.finetune_cnn_after >= 0 and iter >= opt.finetune_cnn_after then
    cnn_grad_params:zero()
  end

  -----------------------------------------------------------------------------
  -- Forward pass
  -----------------------------------------------------------------------------
  -- get batch of data  
  local data = loader:getBatch{batch_size = opt.batch_size, split = 'train', seq_per_img = opt.seq_per_img, distrub_lable = opt.distrub_lable}
  data.images = net_utils.prepro(data.images, true, opt.gpuid >= 0) -- preprocess in place, do data augmentation
  -- data.images: Nx3x224x224 
  -- data.seq: LxM where L is sequence length upper bound, and M = N*seq_per_img

  -- forward the ConvNet on images (most work happens here)
  local feats = protos.cnn:forward(data.images)
  -- we have to expand out image features, once for each sentence
  local baseline_seq = protos.lm:sample(feats, {beam_size=1,sample_max=1})
  local baseline_score = metric:eval(baseline_seq, data.labels)
  -- zeros sth
  local sample_seq, log_prob = protos.lm:sample(feats, {beam_size=1,sample_max=0,temperature=0.1})
  local sample_score = metric:eval(sample_seq, data.labels)
  -------------------------------------------------------------------------------------
  -- local vocab = loader:getVocab()
  -- local baseline_sents = net_utils.decode_sequence(vocab, baseline_seq)
  -- local sample_sents = net_utils.decode_sequence(vocab, sample_seq)
  -- for k=1,#baseline_sents do
  --     print(baseline_sents[k])
  --     print(sample_sents[k])
  -- end
-------------------------------------------------------------------------------------
  local gain = sample_score - baseline_score
  -- print(string.format("%f - %f = %f", sample_score:mean()*100, baseline_score:mean()*100, gain:mean()*100))
  local dlogprobs = policy_grad(gain, sample_seq, log_prob)
  if opt.gpuid >= 0 then dlogprobs = dlogprobs:cuda() end
  
  -----------------------------------------------------------------------------
  -- Backward pass
  -----------------------------------------------------------------------------
  -- backprop language model
  local dfeats, ddummy = unpack(protos.lm:backward({feats, sample_seq}, dlogprobs))
  -- backprop the CNN, but only if we are finetuning
  if opt.finetune_cnn_after >= 0 and iter >= opt.finetune_cnn_after then
    local dx = protos.cnn:backward(data.images, dfeats)
  end

  -- clip gradients
  -- print(string.format('claming %f%% of gradients', 100*torch.mean(torch.gt(torch.abs(grad_params), opt.grad_clip))))
  grad_params:clamp(-opt.grad_clip, opt.grad_clip)

  -- apply L2 regularization
  if opt.cnn_weight_decay > 0 then
    cnn_grad_params:add(opt.cnn_weight_decay, cnn_params)
    -- note: we don't bother adding the l2 loss to the total loss, meh.
    cnn_grad_params:clamp(-opt.grad_clip, opt.grad_clip)
  end
  -----------------------------------------------------------------------------

  -- and lets get out!
  local losses = { total_loss = gain:mean()*100, sample_loss = sample_score:mean() * 100 }
  return losses
end

-------------------------------------------------------------------------------
--   /$$      /$$           /$$                 /$$                                    
--  | $$$    /$$$          |__/                | $$                                    
--  | $$$$  /$$$$  /$$$$$$  /$$ /$$$$$$$       | $$        /$$$$$$   /$$$$$$   /$$$$$$ 
--  | $$ $$/$$ $$ |____  $$| $$| $$__  $$      | $$       /$$__  $$ /$$__  $$ /$$__  $$
--  | $$  $$$| $$  /$$$$$$$| $$| $$  \ $$      | $$      | $$  \ $$| $$  \ $$| $$  \ $$
--  | $$\  $ | $$ /$$__  $$| $$| $$  | $$      | $$      | $$  | $$| $$  | $$| $$  | $$
--  | $$ \/  | $$|  $$$$$$$| $$| $$  | $$      | $$$$$$$$|  $$$$$$/|  $$$$$$/| $$$$$$$/
--  |__/     |__/ \_______/|__/|__/  |__/      |________/ \______/  \______/ | $$____/ 
--                                                                           | $$      
--                                                                           | $$      
--                                                                           |__/   
-------------------------------------------------------------------------------
local loss0
local optim_state = {}
local cnn_optim_state = {}
local loss_history = {}
local val_lang_stats_history = {}
local val_loss_history = {}
local best_score
local smooth_loss = 0
while true do  

  -- eval loss/gradient
  local losses = lossFun()
  if smooth_loss == 0 then smooth_loss = losses.sample_loss end
  smooth_loss = losses.sample_loss * 0.01 + smooth_loss * 0.99
  if iter % opt.losses_log_every == 0 then loss_history[iter] = losses.total_loss end
  print(string.format('iter %5d: %7.3f | avg: %7.3f', iter, losses.total_loss, smooth_loss))

  -- save checkpoint once in a while (or on final iteration)
  if (iter % opt.save_checkpoint_every == 0 or iter == opt.max_iters) then

    -- evaluate the validation performance
    local val_loss, val_predictions, lang_stats = eval_split('val', {val_images_use = opt.val_images_use})
    print('validation loss: ', val_loss)
    print(lang_stats)
    val_loss_history[iter] = val_loss
    if lang_stats then
      val_lang_stats_history[iter] = lang_stats
    end


    -- write a (thin) json report
    local checkpoint = {}
    checkpoint.opt = opt
    checkpoint.iter = iter
    checkpoint.loss_history = loss_history
    checkpoint.val_loss_history = val_loss_history
    checkpoint.val_predictions = val_predictions -- save these too for CIDEr/METEOR/etc eval
    checkpoint.val_lang_stats_history = val_lang_stats_history

    utils.write_json(checkpoint_path .. '.json', checkpoint)
    print('wrote json checkpoint to ' .. checkpoint_path .. '.json')

    -- write the full model checkpoint as well if we did better than ever
    local current_score
    if lang_stats then
      -- use CIDEr score for deciding how well we did
      current_score = lang_stats['CIDEr']
    else
      -- use the (negative) validation loss as a score
      current_score = -val_loss
    end
    if best_score == nil or current_score > best_score then
      best_score = current_score
      if iter > 0 then -- dont save on very first iteration
        -- include the protos (which have weights) and save to file
        local save_protos = {}
        save_protos.lm = thin_lm -- these are shared clones, and point to correct param storage
        save_protos.cnn = thin_cnn
        checkpoint.protos = save_protos
        -- also include the vocabulary mapping so that we can use the checkpoint 
        -- alone to run on arbitrary images without the data loader
        checkpoint.vocab = loader:getVocab()
        torch.save(checkpoint_path .. '.t7', checkpoint)
        print('wrote checkpoint to ' .. checkpoint_path .. '.t7')
      end
    end
  end

  -- decay the learning rate for both LM and CNN
  local learning_rate = opt.learning_rate
  local cnn_learning_rate = opt.cnn_learning_rate
  if iter > opt.learning_rate_decay_start and opt.learning_rate_decay_start >= 0 then
    local frac = (iter - opt.learning_rate_decay_start) / opt.learning_rate_decay_every
    local decay_factor = math.pow(0.5, frac)
    learning_rate = learning_rate * decay_factor -- set the decayed rate
    cnn_learning_rate = cnn_learning_rate * decay_factor
  end

  -- perform a parameter update
  if opt.optim == 'rmsprop' then
    rmsprop(params, grad_params, learning_rate, opt.optim_alpha, opt.optim_epsilon, optim_state)
  elseif opt.optim == 'adagrad' then
    adagrad(params, grad_params, learning_rate, opt.optim_epsilon, optim_state)
  elseif opt.optim == 'sgd' then
    sgd(params, grad_params, opt.learning_rate)
  elseif opt.optim == 'sgdm' then
    sgdm(params, grad_params, learning_rate, opt.optim_alpha, optim_state)
  elseif opt.optim == 'sgdmom' then
    sgdmom(params, grad_params, learning_rate, opt.optim_alpha, optim_state)
  elseif opt.optim == 'adam' then
    adam(params, grad_params, learning_rate, opt.optim_alpha, opt.optim_beta, opt.optim_epsilon, optim_state)
  else
    error('bad option opt.optim')
  end

  -- do a cnn update (if finetuning, and if rnn above us is not warming up right now)
  if opt.finetune_cnn_after >= 0 and iter >= opt.finetune_cnn_after then
    if opt.cnn_optim == 'sgd' then
      sgd(cnn_params, cnn_grad_params, cnn_learning_rate)
    elseif opt.cnn_optim == 'sgdm' then
      sgdm(cnn_params, cnn_grad_params, cnn_learning_rate, opt.cnn_optim_alpha, cnn_optim_state)
    elseif opt.cnn_optim == 'adam' then
      adam(cnn_params, cnn_grad_params, cnn_learning_rate, opt.cnn_optim_alpha, opt.cnn_optim_beta, opt.optim_epsilon, cnn_optim_state)
    else
      error('bad option for opt.cnn_optim')
    end
  end

  -- stopping criterions
  iter = iter + 1
  if iter % 10 == 0 then collectgarbage() end -- good idea to do this once in a while, i think
  if loss0 == nil then loss0 = losses.total_loss end
  -- if losses.total_loss > loss0 * 20 then
  --   print('loss seems to be exploding, quitting.')
  --   break
  -- end
  if opt.max_iters > 0 and iter >= opt.max_iters then break end -- stopping criterion

end
