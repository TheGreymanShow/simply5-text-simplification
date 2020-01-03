from flask import Flask
from flask import request
from flask import render_template
from flask import flash
import script_2

app = Flask(__name__)
app.config.from_object(__name__)
app.config['SECRET_KEY'] = '7d441f27d441f27567d441f2b6176a'

@app.route('/output', methods=['POST'])
def my_form_post():
    text = request.form['text']

    output=script_2.simply5(text)
    flash(output)

    return render_template("output.html")

if __name__ == '__main__':
    app.debug = True
    app.run()
