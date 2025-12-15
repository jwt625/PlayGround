import pandas as pd

# Check documents
print("=" * 80)
print("DOCUMENTS")
print("=" * 80)
df_docs = pd.read_parquet('./christmas/output/documents.parquet')
print(f'Total documents: {len(df_docs)}')
print('\nDocument titles:')
print(df_docs[['title']].to_string())

# Check entities
print("\n" + "=" * 80)
print("ENTITIES")
print("=" * 80)
df_entities = pd.read_parquet('./christmas/output/entities.parquet')
print(f'Total entities: {len(df_entities)}')
print('\nTop entities by degree:')
print(df_entities.nlargest(10, 'degree')[['title', 'type', 'degree']].to_string())

# Check text units
print("\n" + "=" * 80)
print("TEXT UNITS")
print("=" * 80)
df_text = pd.read_parquet('./christmas/output/text_units.parquet')
print(f'Total text units: {len(df_text)}')
print('\nFirst 3 text unit previews:')
for i, row in df_text.head(3).iterrows():
    print(f'\n--- Text Unit {i} ---')
    print(f'Text preview: {row["text"][:150]}...')

# Check relationships
print("\n" + "=" * 80)
print("RELATIONSHIPS")
print("=" * 80)
df_rels = pd.read_parquet('./christmas/output/relationships.parquet')
print(f'Total relationships: {len(df_rels)}')

# Check communities
print("\n" + "=" * 80)
print("COMMUNITIES")
print("=" * 80)
df_comm = pd.read_parquet('./christmas/output/communities.parquet')
print(f'Total communities: {len(df_comm)}')
print('\nCommunity levels:')
print(df_comm['level'].value_counts().sort_index())

