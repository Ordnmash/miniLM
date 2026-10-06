from tool import encode

print("running the tokenizer...")

file = "miniLM/converted_chat_formatted.txt"
data = open(file, "r").read()
#print(tokenSet)
vocabs = list(sorted(set(data)))

stoi = {s:i for i,s in enumerate(vocabs)}
itos = {i:s for i,s in enumerate(vocabs)}

enc_data = encode(data, stoi)

vocabs  = [k for k in stoi.keys()]
targetV = 768


def updateVoc(token:str):
  if token in vocabs:
    print(f"token {token} in vocabs as {stoi[token]}")
    return
  vocabs.append(token)
  i           = len(vocabs)-1
  stoi[token] = i
  itos[i]     = token

def refactor(pair: tuple):
  
  token = ''.join(itos[i] for i in pair)
  updateVoc(token)
  new_data = []
  i        = 0

  while i < len(enc_data) - (len(pair) - 1):
    p = tuple([enc_data[i+j] for j in range(len(pair))])
    if p == pair:
      new_data.append(stoi[token])
      i += len(pair)
    else:
      new_data.append(enc_data[i])
      i += 1

  if i < len(enc_data):
    new_data.extend(enc_data[i:])
  print('len of vocabs =',len(vocabs),'\n')
  return new_data

def graph():
  build = {}
  for i,e in enumerate(enc_data):
    if i < len(enc_data) -1:
      p = (e,enc_data[i+1])
      if p not in build:
        build[p] = 1
      else:
        build[p] += 1

  return build

special_tokens = ['<start_of_turn>user\n','<start_of_turn>model\n','<end_of_turn>\n']
for token in special_tokens:
  intkn = tuple([stoi[x] for x in token])
  enc_data = refactor(intkn)

while len(vocabs) < targetV:
  jsn = graph()
  jsor= sorted(jsn, key=lambda x: jsn[x])
  p   = jsor[-1]
  print(f"merging {p} |{''.join([itos[i] for i in p])}| frequency {jsn[p]}\n")
  enc_data = refactor(p)

with open('/home/ordn/Documents/ordn_projects/miniLM/chat_learnedData.txt', 'w') as f:
  f.write(f"{stoi}")
  f.write("\n")
  f.write(f"{itos}")
  f.write("\n")
  for e in enc_data:
    f.write(f"{e}")
    f.write("\n")
