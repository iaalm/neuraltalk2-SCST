require 'nn'
require 'nngraph'

local GRU = {}
function GRU.gru(input_size, output_size, rnn_size, n, dropout, res_rnn)
  dropout = dropout or 0 

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
    -- the input to this layer
    if L == 1 then 
      x = inputs[1]
      input_size_L = input_size
    else 
      x = outputs[(L-1)*2] 
      if dropout > 0 then x = nn.Dropout(dropout)(x):annotate{name='drop_' .. L} end -- apply dropout, if any
      input_size_L = rnn_size
    end
    -- evaluate the input sums at once for efficiency
    local i2h = nn.Linear(input_size_L, 2 * rnn_size)(x):annotate{name='i2h_'..L}
    local h2h = nn.Linear(rnn_size, 2 * rnn_size)(prev_h):annotate{name='h2h_'..L}
    local all_input_sums = nn.CAddTable()({i2h, h2h})

    local reshaped = nn.Reshape(2, rnn_size)(all_input_sums)
    local n1, n2 = nn.SplitTable(2)(reshaped):split(2)
    -- decode the gates
    local reset_gate = nn.Sigmoid()(n1)
    local update_gate = nn.Sigmoid()(n2)
    local not_update_gate = nn.AddConstant(1,true)(nn.Mul(-1)(update_gate))
    -- decode the write inputs
    local in_transform = nn.CMulTable()({reset_gate, prev_h})
    local i2o = nn.Linear(input_size_L, rnn_size)(x):annotate{name='i2o_'..L}
    local h2o = nn.Linear(rnn_size, rnn_size)(in_transform):annotate{name='h2o_'..L}

    local h_hat = nn.Tanh()(nn.CAddTable()({i2o, h2o}))
    -- perform the LSTM update
    local next_h_raw           = nn.CAddTable()({
        nn.CMulTable()({not_update_gate, prev_h}),
        nn.CMulTable()({update_gate,     h_hat})
      })
    local next_h = nn.Tanh()(next_h_raw)
    
    table.insert(outputs, prev_c)
    if res_rnn > 0 and L > res_rnn and (L - 1) % res_rnn == 0 then
      table.insert(outputs, nn.CAddTable()({next_h, outputs[(L-res_rnn)*2]}))
    else
      table.insert(outputs, next_h)
    end 
  end

  -- set up the decoder
  local top_h = outputs[#outputs]
  if dropout > 0 then top_h = nn.Dropout(dropout)(top_h):annotate{name='drop_final'} end
  local proj = nn.Linear(rnn_size, output_size)(top_h):annotate{name='decoder'}
  local logsoft = nn.LogSoftMax()(proj)
  table.insert(outputs, logsoft)

  return nn.gModule(inputs, outputs)
end

return GRU

