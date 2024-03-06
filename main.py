import sys
from task1 import *
from task2 import *

def main() :
    if len(sys.argv) != 3 : 
        print("Usage: python3 main.py <part id> <input path> <output path>")
    else :
        if int(sys.argv[0]) == 1 : # argv[1] contains 6 images
            task1(sys.argv[1], sys.argv[2])
        elif int(sys.argv[0]) == 2 :
            task1(sys.argv[1], sys.argv[2])
        else : print("Incorrect part id")



if __name__ == "__main__" :
    main()