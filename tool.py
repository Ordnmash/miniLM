import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

def get_data(self, state='train', batch_size=1):
  x, y = [],[]
  encoded_data = self.enctrain if state == 'train' else self.encval

  for _ in range(batch_size):
    inx    = torch.randint(0, (len(encoded_data)-(self.block_size+1)), (1,)).item()
    xx     = encoded_data[inx:inx+self.block_size]
    yy     = encoded_data[inx+1:(inx+self.block_size)+1]
    x.append(xx)
    y.append(yy)

  return (torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long))

# encoding and decoding tools:
# trained tokenizer encoder style:
def tknEncoder(s, stoi, itos):

  voc     = list(reversed(stoi.keys()))
  tokens  = []
  def iterate(x):
    largT = max([sorted(stoi.keys(), key=lambda x: len(x), reverse=True)])
    if len(x) > len(largT): x = x[:len(largT)] # saves computation time.
    while len(x) > 0:
      if x in voc:
        return stoi[x]
      else:
        x = x[:-1]

  while len(s) > 0:
    token = iterate(s)
    tokens.append(token)
    s = s[len(itos[token]):]

  return tokens 


def encode(s: str, stoi) -> list:
  d = []
  for si in s:
    d.append(stoi[si])
  return d

def decode(l: list, itos) -> str:
  d = []
  for li in l:
    d.append(itos[li])
  return ''.join(d)

class LearnedPE(nn.Module):
  def __init__(self, max_seq_len:int, n_embed:int):
    super().__init__()
    self.emb = nn.Embedding(max_seq_len, n_embed)

  def forward(self,x):
    # x = [B, T, C]
    _,T,_   = x.shape
    seq_len = T
    inn     = torch.arange(0, seq_len)
    out     = self.emb(inn)
    return x + out

# this forward pass logic works when nn.MultiheadAttention batch_first = False
def forward(self, x, targets=None):
  xem    = self.embed(x)
  _,T,_  = xem.shape
  mask   = torch.triu(torch.ones(T, T), 1).bool()
  x,_    = self.mha1(xem,xem,xem, attn_mask=mask)
  x      = self.ffn1(x)
  xrb    = x + xem
  _,T,_  = xrb.shape
  mask   = torch.triu(torch.ones(T, T), 1).bool()
  x,_    = self.mha2(xrb,xrb,xrb, attn_mask=mask) 
  x      = self.ffn2(x)
  xrb    = x + xrb
  logits = self.lgts(xrb)
# logits = [B, T, C]

  if targets is not None: # cross_entropy expects [B, C, T]
    logits = logits.transpose(1,2)
    loss   = F.cross_entropy(logits, targets)
  else:
    loss   = None

  return logits, loss

def generate(self, stoi, itos, block_size, use_memory=False):
  self.eval()
  formats = {'userstart' :'\n<start_of_turn>user\n',
           'modelstart':'\n<start_of_turn>model\n',
           'end':'<end_of_turn>'}
  self.chat = []
  while True:
    text  = ''
    human = input("talk to miniLM: ")
    self.chat += [{'you': human, 'ai':''}]

    if human == 'end':
      self.show_chat()
      self.chat = []
      break
    elif human == 'restart':
      self.chat = []
      human = input("talk to minLM: ")
      self.chat += [{'you': human, 'ai':''}]
    
    if use_memory:
      human = []
      for m in self.chat:
        human.append(formats['userstart' ] + m['you'].strip() + formats['end'])
        human.append(formats['modelstart'] + m['ai' ].strip() + formats['end'])
      human = ''.join(human)
      human = human[:-len(formats['end'])]
      human = human if len(human) < block_size else human[len(human)-block_size:]
    else:
      human = formats['userstart'] +human.strip() + formats['end'] + formats['modelstart']

    # start sampling from the model...
    inn = torch.tensor(encode(human, stoi)).unsqueeze(0) # [1,T,C]
    while True:
      logits, _ = self(inn)
      ixlogits  = logits[0, -1]
      probs     = F.softmax(ixlogits, dim=-1)
      ix        = torch.multinomial(probs, num_samples=1).item()

      if formats['end'] not in text:
        lin   = inn.view(-1).tolist(); lin.append(ix)
        if len(lin) > block_size:
          lin = lin[1:]
        inn   = torch.tensor(lin).unsqueeze(0) # [1,T,C]
        text += itos[ix]
      else:
        text  = text.replace("<end_of_turn>", "")
        break
    self.chat[-1]['ai'] = text
    self.show_chat()

def show_chat(self):
  for m in self.chat:
    print('You: ',     m['you'],"\n")
    print('    AI : ', m['ai'], "\n")
  print("##########################")

def fit(self, epochs=1000, batch_size=1, lr=1e-3):
  self.optimizer = optim.AdamW(self.parameters(), lr=lr)
  for i in range(epochs):
    t0 = time.perf_counter()
    self.train()
    self.optimizer.zero_grad(set_to_none=True)

    x, y   = self.get_data('train', batch_size)
    t1 = time.perf_counter()    
    _,loss= self(x, y)
    tforward = time.perf_counter()
    loss.backward()
    t2 = time.perf_counter()
    # update
    self.optimizer.step()
    t3 = time.perf_counter()
    if (i+1) % max(1, int(epochs/20)) == 0:
      # validation
      self.eval()
      with torch.no_grad():
        xv, yv = self.get_data('val',1)
        _, valloss = self(xv, yv)

      print(f"epoch:{i+1}   | train loss={loss.item():.4f}   |  val loss={valloss.item():.4f}")
      print(
        f"  trn constrctn: {t1-t0:.2f}s | "
        f"frwrd: {tforward-t1:.2f}s | "
        f"bckwrd: {t2-tforward:.2f}s | "
        f"updt: {t3-t2:.2f}s"
      )
      torch.save(self.state_dict(), "miniLM.pt") # save checkpoint during training...
