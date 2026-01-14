

# proper plyvel installation

```bash
cd /Users/wentao/Documents/GitHub/PlayGround/20260113_augment_export && source .venv/bin/activate && CXXFLAGS="-I/opt/homebrew/opt/leveldb/include" LDFLAGS="-L/opt/homebrew/opt/leveldb/lib" uv pip install plyvel
```

```bash
cd /Users/wentao/Documents/GitHub/PlayGround/20260113_augment_export && source .venv/bin/activate && CXXFLAGS="-I/opt/homebrew/opt/leveldb/include" LDFLAGS="-L/opt/homebrew/opt/leveldb/lib -Wl,-rpath,/opt/homebrew/opt/leveldb/lib -lleveldb" uv pip install --no-cache-dir plyvel
```


Run script that needs the leveldb library:
```bash
cd /Users/wentao/Documents/GitHub/PlayGround/20260113_augment_export && source .venv/bin/activate && DYLD_LIBRARY_PATH=/opt/homebrew/opt/leveldb/lib python extract_with_leveldb.py
```
