import torch
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import numpy as np
from pathlib import Path
import sqlite3
import os


class ImageMatcher:
    def __init__(self):
        # 保持原有的初始化代码
        self.model = models.resnet50(pretrained=True)
        self.model.eval()
        self.model = torch.nn.Sequential(*list(self.model.children())[:-1])

        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
        ])

        # 设置数据库图片目录
        self.database_dir = 'static/database'

    def _get_db_connection(self):
        """获取数据库连接"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        
        # 确保表存在
        self._ensure_tables_exist(conn)
        return conn
    
    def _ensure_tables_exist(self, conn):
        """确保所有必要的表都存在"""
        cursor = conn.cursor()
        
        # 创建图片主表
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS images (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT NOT NULL,
            path TEXT NOT NULL,
            features BLOB NOT NULL,
            hash_value TEXT,
            file_size INTEGER,
            width INTEGER,
            height INTEGER,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(path)
        )
        ''')

        # 创建图片标签表
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS tags (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL UNIQUE
        )
        ''')

        # 创建图片-标签关联表
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS image_tags (
            image_id INTEGER,
            tag_id INTEGER,
            PRIMARY KEY (image_id, tag_id),
            FOREIGN KEY (image_id) REFERENCES images(id) ON DELETE CASCADE,
            FOREIGN KEY (tag_id) REFERENCES tags(id) ON DELETE CASCADE
        )
        ''')

        # 创建相似度缓存表
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS similarity_cache (
            image1_id INTEGER,
            image2_id INTEGER,
            similarity_score FLOAT,
            calculated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (image1_id, image2_id),
            FOREIGN KEY (image1_id) REFERENCES images(id) ON DELETE CASCADE,
            FOREIGN KEY (image2_id) REFERENCES images(id) ON DELETE CASCADE
        )
        ''')

        # 创建索引
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_images_path ON images(path)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_tags_name ON tags(name)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_images_hash ON images(hash_value)')

        conn.commit()

    def create_database(self):
        """创建并初始化图像数据库"""
        try:
            cursor = self.conn.cursor()

            # 创建图片主表
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS images (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT NOT NULL,
                path TEXT NOT NULL,
                features BLOB NOT NULL,
                hash_value TEXT,           -- 添加这个字段
                file_size INTEGER,
                width INTEGER,
                height INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(path)
            )
            ''')

            # 创建图片标签表
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS tags (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL UNIQUE
            )
            ''')

            # 创建图片-标签关联表
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS image_tags (
                image_id INTEGER,
                tag_id INTEGER,
                PRIMARY KEY (image_id, tag_id),
                FOREIGN KEY (image_id) REFERENCES images(id) ON DELETE CASCADE,
                FOREIGN KEY (tag_id) REFERENCES tags(id) ON DELETE CASCADE
            )
            ''')

            # 创建相似度缓存表
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS similarity_cache (
                image1_id INTEGER,
                image2_id INTEGER,
                similarity_score FLOAT,
                calculated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (image1_id, image2_id),
                FOREIGN KEY (image1_id) REFERENCES images(id) ON DELETE CASCADE,
                FOREIGN KEY (image2_id) REFERENCES images(id) ON DELETE CASCADE
            )
            ''')

            # 创建更新触发器
            cursor.execute('''
            CREATE TRIGGER IF NOT EXISTS update_images_timestamp 
            AFTER UPDATE ON images
            BEGIN
                UPDATE images SET updated_at = CURRENT_TIMESTAMP
                WHERE id = NEW.id;
            END;
            ''')

            # 创建索引以提高查询性能
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_images_path ON images(path)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_tags_name ON tags(name)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_images_hash ON images(hash_value)')

            self.conn.commit()
            print("Database created successfully")

        except Exception as e:
            print(f"Error creating database: {e}")
            self.conn.rollback()
            raise

    def add_image_to_db(self, image_path):
        """添加图片到数据库"""
        try:
            # 打开图片获取基本信息
            with Image.open(image_path) as img:
                width, height = img.size
                file_size = os.path.getsize(image_path)

                # 计算图片特征
                features = self.extract_features(image_path)

                # 计算图片hash值（用于快速查重）
                hash_value = self.calculate_image_hash(img)

                cursor = self.conn.cursor()

                # 检查是否已存在相同路径的图片
                cursor.execute('SELECT id FROM images WHERE path = ?', (str(image_path),))
                existing_image = cursor.execute.fetchone()

                if existing_image:
                    # 更新现有记录
                    cursor.execute('''
                    UPDATE images 
                    SET features = ?, hash_value = ?, file_size = ?, 
                        width = ?, height = ?
                    WHERE path = ?
                    ''', (features.tobytes(), hash_value, file_size,
                          width, height, str(image_path)))
                else:
                    # 插入新记录
                    cursor.execute('''
                    INSERT INTO images (
                        filename, path, features, hash_value, 
                        file_size, width, height
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    ''', (os.path.basename(image_path), str(image_path),
                          features.tobytes(), hash_value, file_size, width, height))

                self.conn.commit()
                return True

        except Exception as e:
            print(f"Error adding image to database: {e}")
            self.conn.rollback()
            return False

    def calculate_image_hash(self, image):
        """计算图片的hash值用于快速查重"""
        # 将图片调整为统一大小
        image = image.resize((8, 8), Image.LANCZOS)
        # 转换为灰度图
        image = image.convert('L')
        # 获取像素数据
        pixels = list(image.getdata())
        # 计算平均值
        avg = sum(pixels) / len(pixels)
        # 生成hash值（1表示大于平均值，0表示小于平均值）
        hash_bits = ''.join(['1' if pixel > avg else '0' for pixel in pixels])
        return hash_bits

    def find_similar_images(self, query_image_path, top_k=5, similarity_threshold=0.5):
        """直接从文件夹读取图片进行相似度计算"""
        try:
            # 提取查询图片的特征
            query_features = self.extract_features(query_image_path)
            
            # 获取数据库目录中的所有图片
            database_images = []
            for ext in ['*.jpg', '*.jpeg', '*.png']:
                database_images.extend(Path(self.database_dir).glob(ext))
            
            print(f"数据库中的图片数量: {len(database_images)}")
            
            similarities = []
            
            # 计算每张图片的相似度
            for img_path in database_images:
                try:
                    # 提取特征
                    features = self.extract_features(str(img_path))
                    
                    # 计算相似度
                    similarity = np.dot(query_features, features) / (
                        np.linalg.norm(query_features) * np.linalg.norm(features)
                    )
                    
                    print(f"图片 {img_path.name} 的相似度: {similarity}")
                    
                    similarities.append({
                        'filename': img_path.name,
                        'path': str(img_path).replace('\\', '/'),  # 确保路径格式正确
                        'similarity': float(similarity)
                    })
                    
                except Exception as e:
                    print(f"处理图片 {img_path} 时出错: {e}")
                    continue
            
            # 按相似度降序排序
            similarities.sort(key=lambda x: x['similarity'], reverse=True)
            
            # 返回相似度最高的图片
            return similarities[:top_k] if similarities else []
            
        except Exception as e:
            print(f"Error finding similar images: {e}")
            print(f"错误详情: {str(e)}")
            import traceback
            print(traceback.format_exc())
            return []

    def get_all_images(self):
        """获取数据库中的所有图片"""
        try:
            conn = self._get_db_connection()
            cursor = conn.cursor()
            
            try:
                cursor.execute('''
                SELECT id, filename, path, file_size, width, height, created_at 
                FROM images 
                ORDER BY created_at DESC
                ''')

                images = []
                for row in cursor.fetchall():
                    images.append({
                        'id': row[0],
                        'filename': row[1],
                        'path': row[2],
                        'file_size': row[3],
                        'width': row[4],
                        'height': row[5],
                        'created_at': row[6]
                    })
                return images
                
            finally:
                conn.close()
                
        except Exception as e:
            print(f"Error getting images: {e}")
            return []

    def delete_image(self, image_id):
        """从数据库中删除图片"""
        try:
            cursor = self.conn.cursor()

            # 首先获取图片路径
            cursor.execute('SELECT path FROM images WHERE id = ?', (image_id,))
            result = cursor.fetchone()

            if result:
                file_path = result[0]

                # 删除数据库记录
                cursor.execute('DELETE FROM images WHERE id = ?', (image_id,))
                self.conn.commit()

                # 删除物理文件
                try:
                    os.remove(file_path)
                except OSError:
                    print(f"Warning: Could not delete file {file_path}")

                return True
            return False
        except Exception as e:
            print(f"Error deleting image: {e}")
            self.conn.rollback()
            return False

    def add_hash_value_column(self):
        """添加 hash_value 列（如果不存在）"""
        try:
            cursor = self.conn.cursor()
            cursor.execute('''
            ALTER TABLE images 
            ADD COLUMN hash_value TEXT
            ''')
            self.conn.commit()
            print("Added hash_value column successfully")
        except Exception as e:
            print(f"Error adding hash_value column: {e}")
            self.conn.rollback()

    def extract_features(self, image_path):
        """提取图片的特征向量"""
        try:
            # 加载图片
            image = Image.open(image_path)
            
            # 如果图片是RGBA格式，转换为RGB
            if image.mode == 'RGBA':
                image = image.convert('RGB')
                
            # 应用预处理变换
            input_tensor = self.transform(image)
            
            # 添加批次维度
            input_batch = input_tensor.unsqueeze(0)
            
            # 使用GPU如果可用
            if torch.cuda.is_available():
                input_batch = input_batch.to('cuda')
                self.model = self.model.to('cuda')

            # 提取特征
            with torch.no_grad():
                features = self.model(input_batch)
                
            # 将特征转换为一维数组
            features = features.squeeze().cpu().numpy()
            
            # 归一化特征向量
            features = features / np.linalg.norm(features)
            
            return features.astype(np.float32)
            
        except Exception as e:
            print(f"Error extracting features from {image_path}: {e}")
            raise