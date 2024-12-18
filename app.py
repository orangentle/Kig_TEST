from flask import Flask, render_template, request, jsonify, redirect, url_for, flash
import os
from werkzeug.utils import secure_filename
from image_processor import ImageMatcher
from pathlib import Path

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['DATABASE_FOLDER'] = 'static/database'  # 存储数据库图片的文件夹
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.secret_key = 'your_secret_key_here'  # 用于flash消息

# 确保必要的文件夹存在
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['DATABASE_FOLDER'], exist_ok=True)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# 初始化图像匹配器
image_matcher = ImageMatcher()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/admin')
def admin():
    # 获取数据库中的所有图片
    db_images = image_matcher.get_all_images()
    return render_template('admin.html', images=db_images)


@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        print(f"上传的文件保存在: {filepath}")

        # 获取最相似的图片
        similar_images = image_matcher.find_similar_images(filepath, top_k=5, similarity_threshold=0.5)
        print(f"找到的相似图片: {similar_images}")
        
        # 构建结果数据
        results = {
            'uploaded_image': {
                'path': f'/static/uploads/{filename}',
                'filename': filename
            },
            'similar_images': similar_images
        }

        return render_template('result.html', results=results)

    return jsonify({'error': 'Invalid file type'}), 400


@app.route('/add-to-database', methods=['POST'])
def add_to_database():
    if 'files[]' not in request.files:
        flash('No file part', 'error')
        return redirect(url_for('admin'))

    files = request.files.getlist('files[]')
    success_count = 0
    error_count = 0

    for file in files:
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['DATABASE_FOLDER'], filename)

            try:
                # 保存文件
                file.save(filepath)
                print(f"文件已保存到: {filepath}")
                
                # 添加到数据库
                if image_matcher.add_image_to_db(filepath):
                    print(f"成功添加图片到数据库: {filename}")
                    success_count += 1
                else:
                    print(f"添加图片到数据库失败: {filename}")
                    error_count += 1
            except Exception as e:
                print(f"处理文件时出错 {filename}: {str(e)}")
                error_count += 1

    if success_count > 0:
        flash(f'Successfully added {success_count} images to database', 'success')
    if error_count > 0:
        flash(f'Failed to add {error_count} images', 'error')

    return redirect(url_for('admin'))


@app.route('/delete-from-database/<int:image_id>', methods=['POST'])
def delete_from_database(image_id):
    if image_matcher.delete_image(image_id):
        flash('Image successfully deleted', 'success')
    else:
        flash('Failed to delete image', 'error')
    return redirect(url_for('admin'))

@app.route('/cleanup-uploads', methods=['POST'])
def cleanup_uploads():
    try:
        shutil.rmtree(app.config['UPLOAD_FOLDER'])
        os.makedirs(app.config['UPLOAD_FOLDER'])
        flash('Upload folder cleaned successfully', 'success')
    except Exception as e:
        flash(f'Error cleaning upload folder: {str(e)}', 'error')
    return redirect(url_for('admin'))

if __name__ == '__main__':
    app.run(debug=True)