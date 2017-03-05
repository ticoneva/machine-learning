import sys
import csv

#Function to save a list or set to a CSV file
def saveCSV(filepath,content):
    try:
        with open(filepath, 'w', newline='') as csvfile:
            mywriter = csv.writer(csvfile)
            mywriter.writerows(content)
            print(filepath,"saved.")
    except FileNotFoundError:
        print("Cannot save to",filepath,". Please check that the filename or filepath is valid")            
    except:
        print("Fail to save", filepath,". Error:",sys.exc_info()[0])


#Function to open a CSV file and load the content to a list
def loadCSV(filepath):
    try:
        with open(filepath, newline='') as csvfile:
            myreader = csv.reader(csvfile)
            mylist = list(myreader)
            print(filepath,"loaded.")
            
    except FileNotFoundError:
        print(filepath,"not found.")
    except:
        print("Fail to load", filepath,". Error:",sys.exc_info()[0])
    
    return mylist       


#Function to open a CSV file and load the content to a 1D list
def loadCSVasList(filepath):

    myList = None

    try:
        with open(filepath, newline='') as csvfile:
            myreader = csv.reader(csvfile)
            myList = [",".join(row) for row in myreader]
            print(filepath,"loaded.")
            
    except FileNotFoundError:
        print(filepath,"not found.")
    except:
        print("Fail to load", filepath,". Error:",sys.exc_info()[0])
    
    return myList


#Function to open a CSV file and load the content to a set
def loadCSVasSet(filepath):
    myList = loadCSVasList(filepath)
    return (set(myList) if myList!=None else None)
    
    
    
