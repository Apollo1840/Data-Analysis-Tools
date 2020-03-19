

import pandas as pd
df=pd.read_excel(r'C:\Users\zouco\Desktop\ALl Collision.xlsx')
df.shape
df.columns
    
df2 = df[(df.ANZKOLL == 1) & (df.VDI1 != 0) & (df.VDI1 != 99) & (df.VDI3 != 0) & (df.VDI2 != 7)]
df2.shape
len(df2.FALL.unique())


# unique_count([1,2,3,4,3,5,6,7,1])

result = df2.groupby('MAIS98')['FALL'].agg('nunique')
result.index
result

# df2.groupby('FALL')['MAIS98'].agg(unique_count).sort_values(ascending=False)


onlyone = lambda x: max(x)
dfr = df2.groupby('FALL')['MAIS98'].agg(onlyone)



df3 = df2[df2.MAIS98 == [dfr[i] for i in df2.FALL]]

result = df3.groupby('MAIS98')['FALL'].agg('nunique')
print(result)
sum(result)

 