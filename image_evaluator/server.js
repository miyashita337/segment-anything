const express = require('express');
const path = require('path');
const fs = require('fs').promises;
const fsSync = require('fs');

const app = express();
const PORT = 3000;

// ミドルウェア設定
app.use(express.json());
app.use(express.static(__dirname));

// 画像ファイル拡張子
const IMAGE_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp'];

/**
 * Windowsパスをサーバー環境用に変換
 */
function convertWindowsPath(windowsPath) {
    // Windowsパス形式（C:/...）をWSL形式（/mnt/c/...）に変換
    if (windowsPath.match(/^[A-Za-z]:/)) {
        const drive = windowsPath.charAt(0).toLowerCase();
        const path = windowsPath.substring(2).replace(/\\/g, '/');
        return `/mnt/${drive}${path}`;
    }
    
    // 既にWSL形式の場合はそのまま
    if (windowsPath.startsWith('/mnt/')) {
        return windowsPath;
    }
    
    // その他の場合は正規化のみ
    return windowsPath.replace(/\\/g, '/');
}

/**
 * ディレクトリ内の画像ファイルを取得
 */
async function getImageFiles(folderPath) {
    try {
        // Windowsパスを適切な形式に変換
        const convertedPath = convertWindowsPath(folderPath);
        console.log(`パス変換: ${folderPath} -> ${convertedPath}`);
        
        // ディレクトリの存在確認
        const stats = await fs.stat(convertedPath);
        if (!stats.isDirectory()) {
            throw new Error(`指定されたパスはディレクトリではありません: ${convertedPath}`);
        }
        
        // ファイル一覧取得
        const files = await fs.readdir(convertedPath);
        
        // 画像ファイルのみフィルタリング
        const imageFiles = files.filter(file => {
            const ext = path.extname(file).toLowerCase();
            return IMAGE_EXTENSIONS.includes(ext);
        });
        
        // ファイルパス付きで返す
        return imageFiles.map(file => ({
            filename: file,
            path: path.join(convertedPath, file)
        }));
        
    } catch (error) {
        console.error(`フォルダー読み込みエラー [${folderPath}]:`, error.message);
        return [];
    }
}

/**
 * 元画像ファイル名から対応する抽出結果を検索
 */
function findCorrespondingImage(originalFilename, extractedImages) {
    // 元ファイル名から拡張子を除去
    const baseName = path.parse(originalFilename).name;
    
    // 抽出結果ファイルの命名パターンを検索
    // 例: "01_kaname03_0000_cover.jpg" -> "01_kaname03_0000_cover_extracted.png"
    const patterns = [
        `${baseName}_extracted`,
        `${baseName}_character`,
        `${baseName}_cropped`,
        `${baseName}`,
        baseName
    ];
    
    for (const pattern of patterns) {
        for (const extractedImage of extractedImages) {
            const extractedBaseName = path.parse(extractedImage.filename).name;
            if (extractedBaseName.includes(pattern) || pattern.includes(extractedBaseName)) {
                return extractedImage.path;
            }
        }
    }
    
    return null;
}

/**
 * フォルダー読み込みAPI
 */
app.post('/api/load-folders', async (req, res) => {
    try {
        const { folder1, folder2 } = req.body;
        
        if (!folder1 || !folder2) {
            return res.json({
                success: false,
                error: '両方のフォルダーパスが必要です'
            });
        }
        
        console.log(`フォルダー読み込み開始:`);
        console.log(`  フォルダー1: ${folder1}`);
        console.log(`  フォルダー2: ${folder2}`);
        
        // 各フォルダーの画像を取得
        const [folder1Images, folder2Images] = await Promise.all([
            getImageFiles(folder1),
            getImageFiles(folder2)
        ]);
        
        console.log(`  フォルダー1画像数: ${folder1Images.length}`);
        console.log(`  フォルダー2画像数: ${folder2Images.length}`);
        
        // フォルダー1をベースとして、対応する画像を検索
        const evaluationData = [];
        
        for (const image1 of folder1Images) {
            const correspondingImage2 = findCorrespondingImage(image1.filename, folder2Images);
            
            evaluationData.push({
                filename: image1.filename,
                folder1_image: image1.path,
                folder2_image: correspondingImage2,
                folder1_rating: '',
                folder2_rating: '',
                folder1_issues_list: [],
                folder2_issues_list: [],
                notes: ''
            });
        }
        
        // フォルダー2にのみ存在する画像も追加
        for (const image2 of folder2Images) {
            const alreadyExists = evaluationData.some(item => 
                findCorrespondingImage(image2.filename, folder1Images)
            );
            
            if (!alreadyExists) {
                evaluationData.push({
                    filename: image2.filename,
                    folder1_image: null,
                    folder2_image: image2.path,
                    folder1_rating: '',
                    folder2_rating: '',
                    folder1_issues_list: [],
                    folder2_issues_list: [],
                    notes: ''
                });
            }
        }
        
        // ファイル名でソート
        evaluationData.sort((a, b) => a.filename.localeCompare(b.filename));
        
        console.log(`  評価対象画像数: ${evaluationData.length}`);
        
        res.json({
            success: true,
            data: evaluationData
        });
        
    } catch (error) {
        console.error('フォルダー読み込みエラー:', error);
        res.json({
            success: false,
            error: error.message
        });
    }
});

/**
 * 画像配信API
 */
app.get('/api/image/:imagePath(*)', (req, res) => {
    try {
        const imagePath = decodeURIComponent(req.params.imagePath);
        
        // Windowsパス変換を適用
        const convertedPath = convertWindowsPath(imagePath);
        
        // パスの正規化とセキュリティチェック
        const normalizedPath = path.resolve(convertedPath);
        
        // ファイルの存在確認
        if (!fsSync.existsSync(normalizedPath)) {
            return res.status(404).send('画像が見つかりません');
        }
        
        // MIMEタイプの設定
        const ext = path.extname(normalizedPath).toLowerCase();
        let contentType = 'image/jpeg'; // デフォルト
        
        switch (ext) {
            case '.png':
                contentType = 'image/png';
                break;
            case '.gif':
                contentType = 'image/gif';
                break;
            case '.bmp':
                contentType = 'image/bmp';
                break;
            case '.webp':
                contentType = 'image/webp';
                break;
        }
        
        res.setHeader('Content-Type', contentType);
        res.setHeader('Cache-Control', 'public, max-age=3600'); // 1時間キャッシュ
        
        // 画像ファイルをストリーミング配信
        const stream = fsSync.createReadStream(normalizedPath);
        stream.pipe(res);
        
        stream.on('error', (error) => {
            console.error('画像読み込みエラー:', error);
            res.status(500).send('画像読み込みエラー');
        });
        
    } catch (error) {
        console.error('画像配信エラー:', error);
        res.status(500).send('内部サーバーエラー');
    }
});

/**
 * ヘルスチェックAPI
 */
app.get('/api/health', (req, res) => {
    res.json({
        status: 'OK',
        timestamp: new Date().toISOString(),
        version: '1.0.0'
    });
});

/**
 * エラーハンドリング
 */
app.use((error, req, res, next) => {
    console.error('サーバーエラー:', error);
    res.status(500).json({
        success: false,
        error: '内部サーバーエラーが発生しました'
    });
});

/**
 * 404ハンドリング
 */
app.use((req, res) => {
    res.status(404).send('ページが見つかりません');
});

/**
 * サーバー起動
 */
app.listen(PORT, '0.0.0.0', () => {
    console.log('🚀 画像評価ツールサーバー起動');
    console.log(`📍 WSLアクセスURL: http://localhost:${PORT}`);
    console.log(`📍 Windowsアクセス: http://127.0.0.1:${PORT}`);
    console.log(`📁 作業ディレクトリ: ${__dirname}`);
    console.log('');
    console.log('🔧 WSL環境での接続方法:');
    console.log('1. Windows側ブラウザで http://127.0.0.1:3000 または http://localhost:3000');
    console.log('2. WSL側で curl http://localhost:3000 でテスト可能');
    console.log('');
    console.log('使用方法:');
    console.log('1. ブラウザで上記URLにアクセス');
    console.log('2. 比較したい2つのフォルダーパスを入力（Windowsパス形式）');
    console.log('3. 「フォルダー読み込み」ボタンをクリック');
    console.log('4. 表示された画像を評価');
    console.log('5. 「CSV出力」で結果をエクスポート');
    console.log('');
    console.log('終了: Ctrl+C');
});

// Graceful shutdown
process.on('SIGINT', () => {
    console.log('\n👋 サーバーを終了中...');
    process.exit(0);
});

process.on('SIGTERM', () => {
    console.log('\n👋 サーバーを終了中...');
    process.exit(0);
});