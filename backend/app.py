from flask import Flask, request, jsonify
from flask_cors import CORS
import mysql.connector
from mysql.connector import Error
import os
from datetime import datetime
import json
import base64
from werkzeug.utils import secure_filename

app = Flask(__name__)
CORS(app)  # Izinkan akses dari frontend Kivy

# ==================== KONFIGURASI DATABASE ====================
DB_CONFIG = {
    'host': 'localhost',
    'user': 'root',
    'password': '',  # Kosong untuk Laragon/XAMPP
    'database': 'hrd_absensi',
    'port': 3306
}

# ==================== KONFIGURASI UPLOAD ====================
UPLOAD_FOLDER = '../uploads'  # Folder untuk menyimpan foto
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Buat folder uploads jika belum ada
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # Maks 16MB

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# ==================== FUNGSI DATABASE ====================
def get_db_connection():
    """Mendapatkan koneksi ke database MySQL"""
    try:
        connection = mysql.connector.connect(**DB_CONFIG)
        return connection
    except Error as e:
        print(f"❌ Error koneksi database: {e}")
        return None

# ==================== API ENDPOINTS ====================

@app.route('/', methods=['GET'])
def home():
    return jsonify({
        'status': 200,
        'message': 'HRD Absensi API is running',
        'endpoints': [
            '/api/v2/face-recognition',
            '/api/v2/face-recognition/register',
            '/api/v2/absen',
            '/api/v2/karyawan/<nrp>',
            '/api/v2/absensi/today',
            '/health'
        ]
    })

@app.route('/health', methods=['GET'])
def health_check():
    """Check kesehatan API dan koneksi database"""
    db_status = 'connected'
    try:
        conn = get_db_connection()
        if conn:
            conn.close()
        else:
            db_status = 'disconnected'
    except:
        db_status = 'error'
    
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'database': db_status
    })

# API 1: GET semua data karyawan untuk face recognition
@app.route('/api/v2/face-recognition', methods=['GET'])
def get_all_karyawan():
    """Mengambil semua data karyawan yang memiliki face encoding"""
    try:
        connection = get_db_connection()
        if not connection:
            return jsonify({'status': 500, 'message': 'Database connection failed'}), 500
        
        cursor = connection.cursor(dictionary=True)
        cursor.execute("""
            SELECT nrp, nama, face_encoding 
            FROM karyawan 
            WHERE face_encoding IS NOT NULL 
            ORDER BY nrp
        """)
        karyawan_list = cursor.fetchall()
        
        cursor.close()
        connection.close()
        
        # Format response
        formatted_data = []
        for karyawan in karyawan_list:
            formatted_data.append({
                'nrp': karyawan['nrp'],
                'name': karyawan['nama'],
                'face_encoding': karyawan['face_encoding']
            })
        
        return jsonify({
            'status': 200,
            'message': 'Success',
            'data': formatted_data
        }), 200
        
    except Exception as e:
        return jsonify({'status': 500, 'message': str(e)}), 500

# API 2: Register face encoding
@app.route('/api/v2/face-recognition/register', methods=['POST'])
def register_face():
    """Menyimpan face encoding ke database"""
    try:
        data = request.get_json()
        
        if not data or not isinstance(data, list) or len(data) == 0:
            return jsonify({'status': 400, 'message': 'Invalid data format'}), 400
        
        karyawan_data = data[0]
        nrp = karyawan_data.get('nrp')
        encodings = karyawan_data.get('encodings')
        
        if not nrp or not encodings:
            return jsonify({'status': 400, 'message': 'NRP and encodings are required'}), 400
        
        # Convert encodings ke JSON string
        face_encoding_json = json.dumps(encodings)
        
        connection = get_db_connection()
        if not connection:
            return jsonify({'status': 500, 'message': 'Database connection failed'}), 500
        
        cursor = connection.cursor(dictionary=True)
        
        # Cek apakah NRP sudah ada
        cursor.execute("SELECT * FROM karyawan WHERE nrp = %s", (nrp,))
        existing = cursor.fetchone()
        
        if existing:
            # Update data yang sudah ada
            cursor.execute("""
                UPDATE karyawan 
                SET face_encoding = %s, updated_at = NOW() 
                WHERE nrp = %s
            """, (face_encoding_json, nrp))
            nama = existing['nama']
        else:
            # Insert data baru
            nama = f"Karyawan {nrp}"
            cursor.execute("""
                INSERT INTO karyawan (nrp, nama, face_encoding) 
                VALUES (%s, %s, %s)
            """, (nrp, nama, face_encoding_json))
        
        connection.commit()
        cursor.close()
        connection.close()
        
        return jsonify({
            'status': 200,
            'message': 'Face registration successful',
            'data': {
                'nrp': nrp,
                'name': nama
            }
        }), 200
        
    except Exception as e:
        return jsonify({'status': 500, 'message': str(e)}), 500

# API 3: Absensi (Check In/Out)
@app.route('/api/v2/absen', methods=['POST'])
def absen():
    """Mencatat absensi karyawan"""
    try:
        # Ambil data dari form-data
        nrp = request.form.get('nrp')
        latitude = request.form.get('latitude')
        longitude = request.form.get('longitude')
        foto = request.files.get('foto')
        
        if not all([nrp, latitude, longitude, foto]):
            return jsonify({'status': 400, 'message': 'Semua field harus diisi'}), 400
        
        # Validasi file foto
        if not allowed_file(foto.filename):
            return jsonify({'status': 400, 'message': 'Format file tidak valid'}), 400
        
        # Simpan foto
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{nrp}_{timestamp}.jpg"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        foto.save(filepath)
        
        connection = get_db_connection()
        if not connection:
            return jsonify({'status': 500, 'message': 'Database connection failed'}), 500
        
        cursor = connection.cursor(dictionary=True)
        
        # Cek data karyawan
        cursor.execute("SELECT nrp, nama FROM karyawan WHERE nrp = %s", (nrp,))
        karyawan = cursor.fetchone()
        
        if not karyawan:
            cursor.close()
            connection.close()
            return jsonify({'status': 404, 'message': 'Karyawan tidak ditemukan'}), 404
        
        nama = karyawan['nama']
        waktu_sekarang = datetime.now()
        tanggal_hari_ini = waktu_sekarang.strftime("%Y-%m-%d")
        
        # Cek absensi hari ini
        cursor.execute("""
            SELECT * FROM absensi 
            WHERE nrp = %s AND DATE(waktu) = %s
            ORDER BY waktu DESC LIMIT 1
        """, (nrp, tanggal_hari_ini))
        
        absensi_terakhir = cursor.fetchone()
        
        # Tentukan status
        if not absensi_terakhir:
            status = "Check In Berhasil"
        else:
            # Cek apakah sudah check out
            waktu_absensi = absensi_terakhir['waktu']
            if isinstance(waktu_absensi, str):
                waktu_absensi = datetime.strptime(waktu_absensi, "%Y-%m-%d %H:%M:%S")
            
            # Jika absensi terakhir adalah check in dan sudah lewat 4 jam
            if "Check In" in absensi_terakhir['status'] and (datetime.now() - waktu_absensi).seconds > 14400:
                status = "Check Out Berhasil"
            else:
                status = f"Sudah {absensi_terakhir['status'].lower()}"
        
        # Simpan data absensi
        cursor.execute("""
            INSERT INTO absensi (nrp, nama, status, latitude, longitude, foto_path)
            VALUES (%s, %s, %s, %s, %s, %s)
        """, (nrp, nama, status, str(latitude), str(longitude), filepath))
        
        connection.commit()
        
        # Ambil data yang baru disimpan
        cursor.execute("""
            SELECT * FROM absensi 
            WHERE id = LAST_INSERT_ID()
        """)
        new_absensi = cursor.fetchone()
        
        cursor.close()
        connection.close()
        
        # Format response
        waktu_str = new_absensi['waktu']
        if isinstance(waktu_str, datetime):
            waktu_str = waktu_str.strftime("%Y-%m-%d %H:%M:%S")
        
        return jsonify({
            'status': 200,
            'message': status,
            'data': {
                'nrp': nrp,
                'name': nama,
                'waktu': waktu_str,
                'status': status,
                'latitude': latitude,
                'longitude': longitude
            }
        }), 200
        
    except Exception as e:
        return jsonify({'status': 500, 'message': str(e)}), 500

# API 4: Get data karyawan by NRP
@app.route('/api/v2/karyawan/<nrp>', methods=['GET'])
def get_karyawan(nrp):
    """Mengambil data karyawan berdasarkan NRP"""
    try:
        connection = get_db_connection()
        if not connection:
            return jsonify({'status': 500, 'message': 'Database connection failed'}), 500
        
        cursor = connection.cursor(dictionary=True)
        cursor.execute("SELECT nrp, nama FROM karyawan WHERE nrp = %s", (nrp,))
        karyawan = cursor.fetchone()
        
        cursor.close()
        connection.close()
        
        if karyawan:
            return jsonify({
                'status': 200,
                'message': 'Success',
                'data': karyawan
            }), 200
        else:
            return jsonify({'status': 404, 'message': 'Karyawan tidak ditemukan'}), 404
            
    except Exception as e:
        return jsonify({'status': 500, 'message': str(e)}), 500

# API 5: Delete karyawan
@app.route('/api/v2/karyawan/<nrp>', methods=['DELETE'])
def delete_karyawan(nrp):
    """Menghapus data karyawan berdasarkan NRP"""
    try:
        connection = get_db_connection()
        if not connection:
            return jsonify({'status': 500, 'message': 'Database connection failed'}), 500
        
        cursor = connection.cursor()
        
        # Hapus data karyawan (ON DELETE CASCADE akan menghapus absensi)
        cursor.execute("DELETE FROM karyawan WHERE nrp = %s", (nrp,))
        connection.commit()
        
        affected_rows = cursor.rowcount
        cursor.close()
        connection.close()
        
        if affected_rows > 0:
            return jsonify({
                'status': 200,
                'message': 'Karyawan berhasil dihapus'
            }), 200
        else:
            return jsonify({'status': 404, 'message': 'Karyawan tidak ditemukan'}), 404
            
    except Exception as e:
        return jsonify({'status': 500, 'message': str(e)}), 500

# API 6: Delete multiple karyawan
@app.route('/api/v2/karyawan', methods=['DELETE'])
def delete_multiple_karyawan():
    """Menghapus multiple data karyawan berdasarkan list NRP"""
    try:
        data = request.get_json()
        nrp_list = data.get('nrp_list', [])
        
        if not nrp_list:
            return jsonify({'status': 400, 'message': 'NRP list tidak boleh kosong'}), 400
        
        connection = get_db_connection()
        if not connection:
            return jsonify({'status': 500, 'message': 'Database connection failed'}), 500
        
        cursor = connection.cursor()
        
        # Format untuk query IN
        placeholders = ', '.join(['%s'] * len(nrp_list))
        query = f"DELETE FROM karyawan WHERE nrp IN ({placeholders})"
        
        cursor.execute(query, nrp_list)
        connection.commit()
        
        deleted_count = cursor.rowcount
        cursor.close()
        connection.close()
        
        return jsonify({
            'status': 200,
            'message': f'{deleted_count} data karyawan berhasil dihapus',
            'deleted_count': deleted_count
        }), 200
        
    except Exception as e:
        return jsonify({'status': 500, 'message': str(e)}), 500

# API 7: Get absensi hari ini
@app.route('/api/v2/absensi/today', methods=['GET'])
def get_absensi_today():
    """Mengambil semua data absensi hari ini"""
    try:
        connection = get_db_connection()
        if not connection:
            return jsonify({'status': 500, 'message': 'Database connection failed'}), 500
        
        cursor = connection.cursor(dictionary=True)
        
        today = datetime.now().strftime("%Y-%m-%d")
        cursor.execute("""
            SELECT * FROM absensi 
            WHERE DATE(waktu) = %s 
            ORDER BY waktu DESC
        """, (today,))
        
        absensi_list = cursor.fetchall()
        
        # Format datetime ke string
        for absensi in absensi_list:
            if isinstance(absensi['waktu'], datetime):
                absensi['waktu'] = absensi['waktu'].strftime("%Y-%m-%d %H:%M:%S")
        
        cursor.close()
        connection.close()
        
        return jsonify({
            'status': 200,
            'message': 'Success',
            'data': absensi_list
        }), 200
        
    except Exception as e:
        return jsonify({'status': 500, 'message': str(e)}), 500

# API 8: Get absensi by NRP
@app.route('/api/v2/absensi/<nrp>', methods=['GET'])
def get_absensi_by_nrp(nrp):
    """Mengambil riwayat absensi karyawan berdasarkan NRP"""
    try:
        connection = get_db_connection()
        if not connection:
            return jsonify({'status': 500, 'message': 'Database connection failed'}), 500
        
        cursor = connection.cursor(dictionary=True)
        
        cursor.execute("""
            SELECT * FROM absensi 
            WHERE nrp = %s 
            ORDER BY waktu DESC
            LIMIT 50
        """, (nrp,))
        
        absensi_list = cursor.fetchall()
        
        # Format datetime ke string
        for absensi in absensi_list:
            if isinstance(absensi['waktu'], datetime):
                absensi['waktu'] = absensi['waktu'].strftime("%Y-%m-%d %H:%M:%S")
        
        cursor.close()
        connection.close()
        
        return jsonify({
            'status': 200,
            'message': 'Success',
            'data': absensi_list
        }), 200
        
    except Exception as e:
        return jsonify({'status': 500, 'message': str(e)}), 500

# API 9: Add karyawan baru (tanpa face encoding)
@app.route('/api/v2/karyawan', methods=['POST'])
def add_karyawan():
    """Menambah data karyawan baru"""
    try:
        data = request.get_json()
        nrp = data.get('nrp')
        nama = data.get('nama')
        
        if not nrp or not nama:
            return jsonify({'status': 400, 'message': 'NRP dan Nama harus diisi'}), 400
        
        if len(nrp) != 10:
            return jsonify({'status': 400, 'message': 'NRP harus 10 digit'}), 400
        
        connection = get_db_connection()
        if not connection:
            return jsonify({'status': 500, 'message': 'Database connection failed'}), 500
        
        cursor = connection.cursor()
        
        try:
            cursor.execute("""
                INSERT INTO karyawan (nrp, nama) 
                VALUES (%s, %s)
            """, (nrp, nama))
            connection.commit()
            
            return jsonify({
                'status': 200,
                'message': 'Karyawan berhasil ditambahkan',
                'data': {'nrp': nrp, 'nama': nama}
            }), 200
            
        except mysql.connector.IntegrityError:
            return jsonify({'status': 400, 'message': 'NRP sudah terdaftar'}), 400
            
        finally:
            cursor.close()
            connection.close()
        
    except Exception as e:
        return jsonify({'status': 500, 'message': str(e)}), 500

# API 10: Update karyawan
@app.route('/api/v2/karyawan/<nrp>', methods=['PUT'])
def update_karyawan(nrp):
    """Update data karyawan"""
    try:
        data = request.get_json()
        nama = data.get('nama')
        
        if not nama:
            return jsonify({'status': 400, 'message': 'Nama harus diisi'}), 400
        
        connection = get_db_connection()
        if not connection:
            return jsonify({'status': 500, 'message': 'Database connection failed'}), 500
        
        cursor = connection.cursor()
        
        cursor.execute("""
            UPDATE karyawan 
            SET nama = %s, updated_at = NOW()
            WHERE nrp = %s
        """, (nama, nrp))
        
        connection.commit()
        affected_rows = cursor.rowcount
        cursor.close()
        connection.close()
        
        if affected_rows > 0:
            return jsonify({
                'status': 200,
                'message': 'Data karyawan berhasil diupdate'
            }), 200
        else:
            return jsonify({'status': 404, 'message': 'Karyawan tidak ditemukan'}), 404
            
    except Exception as e:
        return jsonify({'status': 500, 'message': str(e)}), 500

# ==================== RUN SERVER ====================
if __name__ == '__main__':
    print("=" * 50)
    print("🚀 HRD Absensi API Server")
    print("=" * 50)
    print(f"📁 Upload folder: {UPLOAD_FOLDER}")
    print(f"🗄️  Database: {DB_CONFIG['database']}@{DB_CONFIG['host']}")
    print("\n📡 Endpoints:")
    print("   GET    /                       - Home")
    print("   GET    /health                 - Health check")
    print("   GET    /api/v2/face-recognition - Get all faces")
    print("   POST   /api/v2/face-recognition/register - Register face")
    print("   POST   /api/v2/absen           - Absensi")
    print("   GET    /api/v2/karyawan/<nrp>  - Get karyawan")
    print("   POST   /api/v2/karyawan        - Add karyawan")
    print("   PUT    /api/v2/karyawan/<nrp>  - Update karyawan")
    print("   DELETE /api/v2/karyawan        - Delete multiple")
    print("   DELETE /api/v2/karyawan/<nrp>  - Delete single")
    print("   GET    /api/v2/absensi/today   - Get today's absensi")
    print("   GET    /api/v2/absensi/<nrp>   - Get absensi by NRP")
    print("\n" + "=" * 50)
    print("✅ Server running on http://localhost:5000")
    print("=" * 50)
    
    app.run(host='0.0.0.0', port=5000, debug=True)