import pandas as pd
from scipy import stats

# nivel de expresion en escala logaritmica.
df = pd.read_csv('/home/lobon/Downloads/expression_data_skin_aging.txt', sep='\t')
df = df.set_index('GeneSymbol')
df_y = df[list(filter(lambda x: x.startswith('young'), df.columns))]
df_o = df[list(filter(lambda x: x.startswith('old'), df.columns))]
df['avg_y'] = df_y.mean(axis=1)
df['std_y'] = df_y.std(axis=1)
df['avg_o'] = df_o.mean(axis=1)
df['std_o'] = df_o.std(axis=1)

alpha = 0.05
related = stats.ttest_ind(df_y, df_o, axis=1)
df['related'] = related.pvalue < alpha
df['related'] = df['related'].apply(lambda x: 1 if x else 0)

prot_anno_dat = pd.read_csv('data/protein_annotation.csv')
string_info_dat = pd.read_csv('data/string_info.csv')
string_ppi_dat = pd.read_csv('data/string_ppi.csv')

is_in = list(set(string_info_dat.preferred_name).intersection(df.index.values))

df_info = string_info_dat[['#string_protein_id','preferred_name']].set_index('preferred_name')
merged_df = df_info.join(df).set_index('#string_protein_id')

nodes = string_ppi_dat[['protein1','protein2']]
nodes.set_index('protein1', inplace=True)
nodes = nodes.join(merged_df[['avg_y', 'std_y', 'avg_o', 'std_o', 'related']], how='outer')#.reset_index()
nodes.index.name = 'protein1'
nodes.reset_index(inplace=True)
