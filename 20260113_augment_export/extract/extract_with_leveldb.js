#!/usr/bin/env node
/**
 * Extract Augment conversation histories using proper LevelDB library.
 */

const { ClassicLevel } = require('classic-level');
const fs = require('fs');
const path = require('path');
const os = require('os');

const VSCODE_STORAGE = path.join(os.homedir(), 'Library/Application Support/Code/User/workspaceStorage');

function findWorkspaceFolders() {
    const workspaces = {};
    
    if (!fs.existsSync(VSCODE_STORAGE)) {
        console.error('VSCode storage path not found:', VSCODE_STORAGE);
        return workspaces;
    }
    
    const dirs = fs.readdirSync(VSCODE_STORAGE);
    
    for (const dir of dirs) {
        const workspaceDir = path.join(VSCODE_STORAGE, dir);
        const stat = fs.statSync(workspaceDir);
        
        if (!stat.isDirectory()) continue;
        
        const workspaceJson = path.join(workspaceDir, 'workspace.json');
        if (fs.existsSync(workspaceJson)) {
            try {
                const data = JSON.parse(fs.readFileSync(workspaceJson, 'utf8'));
                let folderPath = data.folder || '';
                
                if (folderPath.startsWith('file://')) {
                    folderPath = folderPath.replace('file://', '');
                }
                
                workspaces[dir] = {
                    path: folderPath,
                    workspaceDir: workspaceDir
                };
            } catch (e) {
                console.error(`Error reading ${workspaceJson}:`, e.message);
            }
        }
    }
    
    return workspaces;
}

async function extractConversationsFromKvStore(kvStorePath) {
    const allData = [];
    
    if (!fs.existsSync(kvStorePath)) {
        return allData;
    }
    
    try {
        const db = new ClassicLevel(kvStorePath, { createIfMissing: false });
        
        for await (const [key, value] of db.iterator()) {
            try {
                const keyStr = key.toString('utf8');
                const valueStr = value.toString('utf8');
                
                try {
                    const obj = JSON.parse(valueStr);
                    // Filter for conversation-related objects
                    if (obj.conversationId || obj.uuid || obj.request_message || 
                        obj.response_text || obj.request_nodes || obj.response_nodes) {
                        allData.push(obj);
                    }
                } catch (e) {
                    // Not JSON, skip
                }
            } catch (e) {
                // Skip problematic entries
            }
        }
        
        await db.close();
    } catch (e) {
        console.error(`  Error opening database: ${e.message}`);
    }
    
    return allData;
}

async function main() {
    console.log('Extracting Augment conversation histories using LevelDB...');
    console.log(`VSCode storage path: ${VSCODE_STORAGE}`);
    console.log();
    
    const workspaces = findWorkspaceFolders();
    console.log(`Found ${Object.keys(workspaces).length} workspaces`);
    console.log();
    
    const outputDir = 'augment_conversations_export_leveldb';
    if (!fs.existsSync(outputDir)) {
        fs.mkdirSync(outputDir);
    }
    
    const summary = [];
    
    for (const [workspaceId, info] of Object.entries(workspaces)) {
        const folderPath = info.path;
        const workspaceDir = info.workspaceDir;
        
        const augmentDir = path.join(workspaceDir, 'Augment.vscode-augment');
        if (!fs.existsSync(augmentDir)) continue;
        
        const kvStore = path.join(augmentDir, 'augment-kv-store');
        
        console.log(`Processing: ${folderPath}`);
        console.log(`  Workspace ID: ${workspaceId}`);
        
        const allData = await extractConversationsFromKvStore(kvStore);
        
        if (allData.length > 0) {
            // Organize data by conversation ID
            const conversationsById = {};
            const metadataItems = [];
            
            for (const item of allData) {
                const convId = item.conversationId;
                if (convId) {
                    if (!conversationsById[convId]) {
                        conversationsById[convId] = {
                            conversation_id: convId,
                            exchanges: [],
                            metadata: []
                        };
                    }
                    
                    if (item.request_message !== undefined || item.response_text !== undefined) {
                        conversationsById[convId].exchanges.push(item);
                    } else {
                        conversationsById[convId].metadata.push(item);
                    }
                } else {
                    metadataItems.push(item);
                }
            }
            
            const safeName = folderPath.replace(/\//g, '_').replace(/:/g, '_');
            const outputFile = path.join(outputDir, `${workspaceId}_${safeName}.json`);
            
            const conversations = Object.values(conversationsById);
            const totalExchanges = conversations.reduce((sum, c) => sum + c.exchanges.length, 0);
            
            fs.writeFileSync(outputFile, JSON.stringify({
                workspace_id: workspaceId,
                folder_path: folderPath,
                extracted_at: new Date().toISOString(),
                total_items: allData.length,
                conversation_count: conversations.length,
                conversations: conversations,
                other_items: metadataItems
            }, null, 2));
            
            console.log(`  Extracted ${conversations.length} conversations with ${totalExchanges} exchanges`);
            console.log(`  Saved to: ${outputFile}`);
            
            summary.push({
                workspace_id: workspaceId,
                folder_path: folderPath,
                conversation_count: conversations.length,
                exchange_count: totalExchanges,
                output_file: outputFile
            });
        } else {
            console.log('  No conversations found');
        }
        
        console.log();
    }
    
    // Save summary
    const summaryFile = path.join(outputDir, 'extraction_summary.json');
    fs.writeFileSync(summaryFile, JSON.stringify({
        extracted_at: new Date().toISOString(),
        total_workspaces: summary.length,
        workspaces: summary
    }, null, 2));
    
    console.log('\nExtraction complete!');
    console.log(`Summary saved to: ${summaryFile}`);
    console.log(`Total workspaces with conversations: ${summary.length}`);
}

main().catch(console.error);

