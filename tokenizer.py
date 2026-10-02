from tool import encode

file = "miniLM/tokenization_properties.txt"
tokenSet = open(file, "r").read().splitlines()
#print(tokenSet)

import ast
encdec = []
for line in tokenSet[:2]:
  #print("converting line to dict: ")
  encdec.append(ast.literal_eval(line))

stoi = encdec[0]
itos = encdec[1]
enc_data = [int(x) for x in tokenSet[2:]]

vocabs  = [k for k in stoi.keys()]
targetV = 500 if len(vocabs) < 500 else len(vocabs)+1


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

  while i < len(enc_data) - 1:
    p = tuple([enc_data[i+j] for j in range(len(pair))])
    if p == pair:
      new_data.append(stoi[token])
      i += len(pair)
    else:
      new_data.append(enc_data[i])
      i += 1

  if i < len(enc_data):
    new_data.append(enc_data[i])
  print('len of vocabs = ',len(vocabs))
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

#print(f"len of vocabs = {len(vocabs)}, targetV = {targetV}")
while len(vocabs) < targetV:
  jsn = graph()
  p   = sorted(jsn, key=lambda x: jsn[x])[-1]
  print(f"{''.join([itos[i] for i in p])}|")
  enc_data = refactor(p)

with open('miniLM/tokenization_properties2.txt', 'w') as f:
  f.write(f"{stoi}")
  f.write("\n")
  f.write(f"{itos}")
  f.write("\n")
  for e in enc_data:
    f.write(f"{e}")
    f.write("\n")
