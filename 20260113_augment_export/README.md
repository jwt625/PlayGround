# Git Commit Rules

NEVER use `git add -A` or `git add .` from a subdirectory - it stages files across the ENTIRE repository, not just the current folder.

ALWAYS:
1. Stage specific files explicitly: `git add path/to/file1 path/to/file2`
2. Verify staged files with `git status` BEFORE committing
3. Before `git reset --hard`, check what files are in the commit with `git show --name-status` - reset deletes untracked files that were in that commit

---

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
