from flask import Flask, render_template, jsonify
import qrcode

app = Flask(__name__)

stats = {
    'organik': 0,
    'nonorganik': 0
}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/reset')
def reset_stats():
    global stats
    stats = {
        'organik': 0,
        'nonorganik': 0
    }
    return jsonify(success=True)

@app.route('/stats')
def get_stats():
    return jsonify(stats)

@app.route('/qrcode')
def generate_qrcode():
    global stats
    qr = qrcode.QRCode(version=1, box_size=10, border=5)
    qr.add_data(f"Organik: {stats['organik']}\nNon-organik: {stats['nonorganik']}")
    qr.make(fit=True)
    img = qr.make_image(fill='black', back_color='white')
    img.save('qrcode.png')
    return jsonify(success=True)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
