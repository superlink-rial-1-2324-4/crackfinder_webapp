import os
import pandas as pd

# functions related to consolidation of data for the home page

def clear_uploads():
    for file in os.listdir('uploads'):
        os.remove(os.path.join('uploads', file))
    return

def access_masterlist():
    masterlist = pd.read_csv('sessions.csv', sep='|')
    masterlist = masterlist.sort_values(by='DateYMD', ascending=False)
    return masterlist

def get_ids():
    masterlist = access_masterlist()
    ids = masterlist['ID'].tolist()
    return ids

def get_dates():
    masterlist = access_masterlist()
    dates = masterlist['DateYMD'].tolist()
    return dates

def get_venues():
    masterlist = access_masterlist()
    venues = masterlist['Venue'].tolist()
    return venues

def get_crackcounts():
    masterlist = access_masterlist()
    crackcounts = masterlist['CrackCount'].tolist()
    return crackcounts

def get_notes():
    masterlist = access_masterlist()
    notes = masterlist['Notes'].tolist()
    return notes

def get_surveycsvpaths():
    masterlist = access_masterlist()
    surveycsvpaths = masterlist['CsvFilename'].tolist()
    return surveycsvpaths

def get_mappaths():
    masterlist = access_masterlist()
    mappaths = masterlist['MapFilename'].tolist()
    return mappaths

# functions to handle the edits made by user to the session details

def update_masterlist(masterlist):
    masterlist.to_csv('sessions.csv', sep = '|', index=False)
    return

def update_venues(new_venue, index):
    masterlist = access_masterlist()
    venues = get_venues()
    if new_venue[-1] != '.': new_venue = new_venue + '.'
    venues[index] = new_venue
    masterlist['Venue'] = venues
    update_masterlist(masterlist)
    return

def update_notes(new_notes, index):
    masterlist = access_masterlist()
    notes = get_notes()
    if new_notes[-1] != '.': new_notes = new_notes + '.'
    notes[index] = new_notes
    masterlist['Notes'] = notes
    update_masterlist(masterlist)
    return

# functions for processing data for high-level overview

def count_sessions():
    sessions = get_dates()
    countsessions = len(sessions)
    return countsessions

def sum_crackcounts():
    crackcounts = get_crackcounts()
    cracksum = sum(crackcounts)
    return cracksum

def get_latestdate():
    dates = get_dates()
    dates.sort(reverse=True)
    if dates:
        latestdate = dates[0]
        return latestdate
    return

# functions to fetch data related to the specific session

# fetch data related to a specific survey session

def access_surveydata(csv_path):
    surveydata = pd.read_csv(csv_path, sep='|')
    return surveydata

def get_classifications(csv_path):
    surveydata = access_surveydata(csv_path)
    classifications = surveydata['Classification'].tolist()
    return classifications

def get_imagepaths(csv_path):
    surveydata = access_surveydata(csv_path)
    imagepaths = surveydata['ImagePath'].tolist()
    return imagepaths

def get_gridlabels(csv_path):
    surveydata = access_surveydata(csv_path)
    gridlabels = surveydata['GridLabel'].tolist()
    return gridlabels

def get_positions(csv_path):
    surveydata = access_surveydata(csv_path)
    positions = surveydata['Position'].tolist()
    return positions

def get_shows(csv_path):
    surveydata = access_surveydata(csv_path)
    shows = surveydata['BooleanShow'].tolist()
    return shows

def get_engineernotes(csv_path):
    surveydata = access_surveydata(csv_path)
    engineernotes = surveydata['Notes'].tolist()
    return engineernotes

def get_griddinfo(id):
    masterlist = access_masterlist()
    numrows = masterlist.loc[masterlist['ID'] == id, 'Rows'].values[0]
    numcols = masterlist.loc[masterlist['ID'] == id, 'Cols'].values[0]
    area = numrows * numcols
    return numrows, numcols, area

# functions update data related to specific survey session

def update_surveydata(csv_path, surveydata):
    surveydata = surveydata.sort_values(by='GridLabel', ascending=True)
    surveydata.to_csv(csv_path, sep = '|', index=False)
    return

def update_engineernotes(csv_path, new_notes, index):
    surveydata = access_surveydata(csv_path)
    notes = get_engineernotes(csv_path)
    if new_notes[-1] != '.': new_notes = new_notes + '.'
    notes[index] = new_notes
    surveydata['Notes'] = notes
    update_surveydata(csv_path, surveydata)
    return