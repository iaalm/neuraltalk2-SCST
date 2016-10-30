require 'nn'
require 'nngraph'

local LSTM = {}
function LSTM.lstm(input_size, output_size, rnn_size, n, dropout_l, dropout_t, res_rnn, normalize, slstm, f_bias)
  dropout_l = dropout_l or 0 
  dropout_t = dropout_t or 0 

  -- there will be 2*n+1 inputs
  local inputs = {}
  table.insert(inputs, nn.Identity()()) -- indices giving the sequence of symbols
  for L = 1,n do
    table.insert(inputs, nn.Identity()()) -- prev_c[L]
    table.insert(inputs, nn.Identity()()) -- prev_h[L]
  end

  local x, input_size_L
  local outputs = {}
  for L = 1,n do
    -- c,h from previos timesteps
    local prev_h = inputs[L*2+1]
    local prev_c = inputs[L*2]
    if dropout_t > 0 then prev_h = nn.Dropout(dropout_t)(prev_h):annotate{name='drop_t_' .. L} end -- apply dropout_t, if any
    -- the input to this layer
    if L == 1 then 
      x = inputs[1]
      input_size_L = input_size
    else 
      if res_rnn > 0 and L > res_rnn + 1 and (L - 2) % res_rnn == 0 then    
        x = nn.CAddTable()({outputs[(L-1)*2], outputs[(L-1-res_rnn)*2]})    
      else
        x = outputs[(L-1)*2] 
      end
      if dropout_l > 0 then x = nn.Dropout(dropout_l)(x):annotate{name='drop_l_' .. L} end -- apply dropout_l, if any
      input_size_L = rnn_size
    end
    -- evaluate the input sums at once for efficiency
    local i2h = nn.Linear(input_size_L, 4 * rnn_size)(x):annotate{name='i2h_'..L}
    local h2h = nn.Linear(rnn_size, 4 * rnn_size)(prev_h):annotate{name='h2h_'..L}
    local all_input_sums = nn.CAddTable()({i2h, h2h})

    local reshaped = nn.Reshape(4, rnn_size)(all_input_sums)
    local n1, n2, n3, n4 = nn.SplitTable(2)(reshaped):split(4)
    n2 = nn.AddConstant(f_bias,true)(n2)
    -- decode the gates
    local in_gate = nn.Sigmoid()(n1)
    local forget_gate = nn.Sigmoid()(n2)
    local out_gate = nn.Sigmoid()(n3)
    -- decode the write inputs
    
    local in_transform
    if slstm then
      in_transform = nn.HardTanh(-1,1,true)(n4)
    else
      in_transform = nn.Tanh()(n4)
    end
    -- perform the LSTM update
    local next_c           = nn.CAddTable()({
        nn.CMulTable()({forget_gate, prev_c}),
        nn.CMulTable()({in_gate,     in_transform})
      })
    -- gated cells form the output
    local next_h
    if slstm then
      next_h = nn.CMulTable()({out_gate, nn.HardTanh()(next_c)})
    else
      next_h = nn.CMulTable()({out_gate, nn.Tanh()(next_c)})
    end
    
    table.insert(outputs, next_c)
    if normalize ~= 0 then
      next_h = nn.Normalize(normalize)(next_h)
    end
    table.insert(outputs, next_h)
  end

  -- set up the decoder
  local top_h = outputs[#outputs]
  if dropout_l > 0 then top_h = nn.Dropout(dropout_l)(top_h):annotate{name='drop_final'} end
  local proj = nn.Linear(rnn_size, output_size)(top_h):annotate{name='decoder'}
  local logsoft = nn.LogSoftMax()(proj)
  table.insert(outputs, logsoft)

  return nn.gModule(inputs, outputs)
end

function LSTM.clstm(input_size, output_size, rnn_size, n, dropout_l, dropout_t, res_rnn, normalize, slstm, f_bias)
  dropout_l = dropout_l or 0 
  dropout_t = dropout_t or 0 

  -- there will be 2*n+1 inputs
  local inputs = {}
  table.insert(inputs, nn.Identity()()) -- indices giving the sequence of symbols
  for L = 1,n do
    table.insert(inputs, nn.Identity()()) -- prev_c[L]
    table.insert(inputs, nn.Identity()()) -- prev_h[L]
  end

  local x, input_size_L
  local outputs = {}
  for L = 1,n do
    -- c,h from previos timesteps
    local prev_h = inputs[L*2+1]
    local prev_c = inputs[L*2]
    if dropout_t > 0 then prev_h = nn.Dropout(dropout_t)(prev_h):annotate{name='drop_t_' .. L} end -- apply dropout_t, if any
    -- the input to this layer
    if L == 1 then 
      x = inputs[1]
      input_size_L = input_size
    else 
      if res_rnn > 0 and L > res_rnn + 1 and (L - 2) % res_rnn == 0 then    
        x = nn.CAddTable()({outputs[(L-1)*2], outputs[(L-1-res_rnn)*2]})    
      else
        x = outputs[(L-1)*2] 
      end
      if dropout_l > 0 then x = nn.Dropout(dropout_l)(x):annotate{name='drop_l_' .. L} end -- apply dropout_l, if any
      input_size_L = rnn_size
    end
    -- evaluate the input sums at once for efficiency
    local i2h = nn.Linear(input_size_L, 3 * rnn_size)(x):annotate{name='i2h_'..L}
    local h2h = nn.Linear(rnn_size, 3 * rnn_size)(prev_h):annotate{name='h2h_'..L}
    local all_input_sums = nn.CAddTable()({i2h, h2h})

    local reshaped = nn.Reshape(3, rnn_size)(all_input_sums)
    local n1, n2, n3 = nn.SplitTable(2)(reshaped):split(3)
    n1 = nn.AddConstant(f_bias,true)(n1)
    -- decode the gates
    local forget_gate = nn.HardTanh(0,1,true)(n1)
    local in_gate = forget_gate - nn.MulConstant(-1) - nn.AddConstant(1)
    local out_gate = nn.HardTanh(0,1,true)(n2)
    -- decode the write inputs
    
    local in_transform
    if slstm then
      in_transform = nn.HardTanh(-1,1,true)(n3)
    else
      in_transform = nn.Tanh()(n3)
    end
    -- perform the LSTM update
    local next_c           = nn.CAddTable()({
        nn.CMulTable()({forget_gate, prev_c}),
        nn.CMulTable()({in_gate,     in_transform})
      })
    -- gated cells form the output
    local next_h
    if slstm then
      next_h = nn.CMulTable()({out_gate, nn.HardTanh()(next_c)})
    else
      next_h = nn.CMulTable()({out_gate, nn.Tanh()(next_c)})
    end
    
    table.insert(outputs, next_c)
    if normalize ~= 0 then
      next_h = nn.Normalize(normalize)(next_h)
    end
    table.insert(outputs, next_h)
  end

  -- set up the decoder
  local top_h = outputs[#outputs]
  if dropout_l > 0 then top_h = nn.Dropout(dropout_l)(top_h):annotate{name='drop_final'} end
  local proj = nn.Linear(rnn_size, output_size)(top_h):annotate{name='decoder'}
  local logsoft = nn.LogSoftMax()(proj)
  table.insert(outputs, logsoft)

  return nn.gModule(inputs, outputs)
end

return LSTM

