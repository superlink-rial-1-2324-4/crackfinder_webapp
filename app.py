from flask import Flask, render_template, request, redirect, url_for, flash

from backend import *
from consolidator import *

app = Flask(__name__)

sessioncsv = ''
sessionmap = ''
sessionid = ''
sessionvenue = ''
sessiondate = ''
show_negative = 1

UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/', methods=['GET', 'POST'])
@app.route('/home', methods=['GET', 'POST'])
def home():
    clear_uploads()
    ids = get_ids()
    dates = get_dates()
    venues = get_venues()
    crackcounts = get_crackcounts()
    notes = get_notes()
    surveycsvpaths = get_surveycsvpaths()
    mappaths = get_mappaths()
    countsessions = count_sessions()
    cracksum = sum_crackcounts()
    latestdate = get_latestdate()

    if request.method == 'POST':
        index = int(request.form.get('index', 0))
        new_venue = request.form.get('new_venue', '')
        new_notes = request.form.get('new_notes', '')
        new_csvpt = request.form.get('sessioncsvpath', '')
        sel_venue = request.form.get('sessionvenue', '')
        sel_date = request.form.get('sessiondate', '')
        uploaded_files = request.form.get('sessionFiles', '')
        if new_csvpt:
            global sessioncsv, sessionid, sessionmap, sessionvenue, sessiondate
            sessioncsv = new_csvpt
            sessionvenue = sel_venue
            sessiondate = sel_date
            sessionid = ids[surveycsvpaths.index(sessioncsv)]
            sessionmap = mappaths[surveycsvpaths.index(sessioncsv)].replace('\\', '\\\\')
            return redirect(url_for('session', sessionid=sessionid, sessioncsvpath=sessioncsv, sessionmap=sessionmap, sessionvenue=sessionvenue, sessiondate=sessiondate, show_negative=show_negative))
        if new_venue:
            update_venues(new_venue, index)
        elif new_notes:
            update_notes(new_notes, index)
        elif uploaded_files:
            None
        return redirect(url_for('home'))
    return render_template("home.html",
                           dates = dates,
                           venues = venues,
                           crackcounts = crackcounts,
                           notes = notes,
                           surveycsvpaths = surveycsvpaths,
                           countsessions = countsessions,
                           cracksum = cracksum,
                           latestdate = latestdate)

@app.route('/upload_session', methods=['POST'])
def upload_session():

    zipfiles = request.files.getlist('zipfile')
    csvfiles = request.files.getlist('csvfile')

    # Check if any files were uploaded
    if not zipfiles and not csvfiles:
        flash('No files were uploaded')
        return redirect(url_for('home'))

    # Save zip files
    for zipfile in zipfiles:
        if zipfile:
            zipfile.save(os.path.join(app.config['UPLOAD_FOLDER'], zipfile.filename))

    # Save csv files
    for csvfile in csvfiles:
        if csvfile:
            csvfile.save(os.path.join(app.config['UPLOAD_FOLDER'], csvfile.filename))
    
    CrackGPT()
    
    return redirect(url_for('home'))

@app.route('/session', methods=['GET', 'POST'])
def session():
    sessioncsvpath = request.args.get('sessioncsvpath')
    sessionmap = request.args.get('sessionmap')
    sessionid = request.args.get('sessionid')
    sessionvenue = request.args.get('sessionvenue')
    sessiondate = request.args.get('sessiondate')
    show_negative = request.args.get('show_negative')
    classifications = get_classifications(sessioncsvpath)
    imagepaths = get_imagepaths(sessioncsvpath)
    gridlabels = get_gridlabels(sessioncsvpath)
    positions = get_positions(sessioncsvpath)
    flagcolors = get_flagcolors(sessioncsvpath)
    shows = get_shows(sessioncsvpath)
    notes = get_engineernotes(sessioncsvpath)
    numrows, numcols, area = get_griddinfo(sessionid)

    if request.method == 'POST':
        index = int(request.form.get('index', 0))
        new_notes = request.form.get('new_notes', '')
        showneg = request.form.get('showneg', '')
        if new_notes:
            update_engineernotes(csv_path = sessioncsvpath, new_notes = new_notes, index = index)
        elif showneg:
            show_negative = showneg
        return redirect(url_for('session',  sessionid=sessionid, sessioncsvpath=sessioncsvpath, sessionmap=sessionmap, show_negative=showneg))

    return render_template("session.html", 
                           sessioncsvpath = sessioncsvpath,
                           sessionmap = sessionmap,
                           sessionid = sessionid,
                           sessionvenue = sessionvenue,
                           sessiondate = sessiondate,
                           show_negative=show_negative,
                           classifications = classifications,
                           imagepaths = imagepaths,
                           gridlabels = gridlabels,
                           positions = positions,
                           flagcolors = flagcolors,
                           shows = shows,
                           notes = notes, 
                           numrows = numrows,
                           numcols = numcols,
                           area = area)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8080)
    #app.run(debug=True, host='127.0.0.1', port=5000)