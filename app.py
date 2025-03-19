from flask import Flask, redirect, render_template, request
from Database import *
from algorithm import allAlgorithms


app = Flask(__name__,
            static_url_path='/assets',
            static_folder='./templates/assets',
            )


@app.route('/logOut')
def logOut():
    return render_template('index.html')


@app.route('/registration' ,  methods=['POST', 'GET'])
def registration():
    if request.method=="POST":
        username = request.form["username"]
        email = request.form["email"]
        password = request.form["password"]
        mobile = request.form["mobile"]
        InsertData(username,email,password,mobile)
        return render_template('login.html')

    return render_template('registration.html')

@app.route('/', methods=['POST', 'GET'])
def login():
    if request.method=="POST":
        email = request.form['username']
        passw = request.form['password']
        resp = read_cred(email, passw)
        if resp != None:
            return redirect("/home")
        else:
            message = "Username and/or Password incorrect.\\n        Yo have not registered Yet \\nGo to Register page and do Registration";
            return "<script type='text/javascript'>alert('{}');</script>".format(message)

    return render_template('index.html')


@app.route('/home')
def home():
    # return render_template('index.html')
    return render_template('home.html')


@app.route('/nltk',methods=["POST"])
def nltk():


    folder = os.getcwd()+"\\"+"assignments"
    perc = request.form['nltk']

    print(perc)

    files = [folder+"\\"+doc for doc in os.listdir(folder) if doc.endswith('.txt')]

    all_files_data = []
    for File in files:
        all_files_data.append(open(File).read())

    summarized_data = []  
    for data in all_files_data:
        summarized_data.append(str(allAlgorithms.nltk_summarize(data,perc)).replace("</s> %",''))
    
    combined_files_data = ''
    for i in summarized_data:
        combined_files_data = combined_files_data +"\n"+ i

    for i in range(len(all_files_data)):
        print('\n\n\nActual\n',all_files_data[i],'\n\nSummarized\n',summarized_data[i])


    print('\n\nCombined Data\n')
    print(combined_files_data)
    print('\n\nCombined Summary\n')
    combine_summary = str(allAlgorithms.nltk_summarize(combined_files_data,perc)).replace("</s> %",'')
    print(combine_summary)

    result = []
    for i in range(len(files)):
        result.append([files[i], all_files_data[i],summarized_data[i]])

    return render_template('result.html',combine_summary=combine_summary,result=result,algorithm = 'NLTK Algorithm')



@app.route('/textRank',methods=["POST"])
def textRank():


    folder = os.getcwd()+"\\"+"assignments"
    perc = request.form['textRank']

    print(perc)

    files = [folder+"\\"+doc for doc in os.listdir(folder) if doc.endswith('.txt')]

    all_files_data = []
    for File in files:
        all_files_data.append(open(File).read())

    summarized_data = []  
    for data in all_files_data:
        summarized_data.append(str(allAlgorithms.spacy_summarize(data,perc)).replace("</s> %",''))
    
    combined_files_data = ''
    for i in summarized_data:
        combined_files_data = combined_files_data +"\n"+ i

    for i in range(len(all_files_data)):
        print('\n\n\nActual\n',all_files_data[i],'\n\nSummarized\n',summarized_data[i])


    print('\n\nCombined Data\n')
    print(combined_files_data)
    print('\n\nCombined Summary\n')
    combine_summary = str(allAlgorithms.spacy_summarize(combined_files_data,perc)).replace("</s> %",'')
    print(combine_summary)

    result = []
    for i in range(len(files)):
        result.append([files[i], all_files_data[i],summarized_data[i]])

    return render_template('result.html',combine_summary=combine_summary,result=result,algorithm = 'Text Rank Algorithm')





@app.route('/transformer',methods=["POST"])
def transformer():


    folder = os.getcwd()+"\\"+"assignments"
    perc = request.form['transformer']

    print(perc)

    files = [folder+"\\"+doc for doc in os.listdir(folder) if doc.endswith('.txt')]

    all_files_data = []
    for File in files:
        all_files_data.append(open(File).read())

    summarized_data = []  
    for data in all_files_data:
        summarized_data.append(str(allAlgorithms.hugging_face_transformer(data)).replace("</s> %",''))
    
    combined_files_data = ''
    for i in summarized_data:
        combined_files_data = combined_files_data +"\n"+ i

    for i in range(len(all_files_data)):
        print('\n\n\nActual\n',all_files_data[i],'\n\nSummarized\n',summarized_data[i])


    print('\n\nCombined Data\n')
    print(combined_files_data)
    print('\n\nCombined Summary\n')
    combine_summary = str(allAlgorithms.hugging_face_transformer(combined_files_data)).replace("</s> %",'')
    print(combine_summary)

    result = []
    for i in range(len(files)):
        result.append([files[i], all_files_data[i],summarized_data[i]])

    return render_template('result.html',combine_summary=combine_summary,result=result,algorithm = 'Transformer Hugging Face Algorithm')



@app.route('/index')
def ai_engine_page():
    return render_template('home.html')


    
@app.route('/singleSum')
def singleSum():
    return render_template('singleSum.html')


@app.route('/output',methods=['POST'])
def output():
    if request.method == 'POST':
        textvalue = request.form.get("textarea", None)
        return render_template('output.html', res=allAlgorithms.custom_trained_model(textvalue))


if __name__ == '__main__':
    app.run(debug=False)