import pandas as pd


def main():
  file = input()
  func = input()

  df = pd.read_csv(file)

  if func == 'Q1':
      print(df.shape)
  elif func == 'Q2':
      print(int(df['score'].max()))
  elif func == 'Q3':
      result = df[df['score'] >= 80]
      print(len(result) if len(result) > 0 else 'No Output')
  elif func == 'Q4':
      result = df[df['score'] >= 80]
      print('No Output')
  else:
      # Do something
      pass

if __name__ == "__main__":
  main()
