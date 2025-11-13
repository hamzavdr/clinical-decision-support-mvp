import chromadb
from chromadb.config import Settings

client = chromadb.PersistentClient(path='./chroma_db', settings=Settings(anonymized_telemetry=False))
coll = client.get_collection('guidelines')

# Get first 5 items with FULL metadata
results = coll.get(limit=5, include=['metadatas', 'documents'])
print(f'Total documents: {coll.count()}')
print('\nFirst 5 documents:')
for i, (doc_id, meta, doc) in enumerate(zip(results['ids'], results['metadatas'], results['documents'])):
    print(f'\n--- Document {i+1} ---')
    print(f'ID: {doc_id}')
    print(f'Metadata: {meta}')
    print(f'Text preview: {doc[:100]}...')