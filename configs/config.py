import configparser 
#from argumentParser import argumentParser
config = None
#args = argumentParser()
import os


def read():
    
    
    # path = os.path.join(my_path, "../data/test.csv")
    # print("dfef",os.getcwd()) 
    # os.chdir("../utils")
    # sys.path.append(os.path.join(os.path.dirname(sys.path[0]),'utils'))
    user = "udi"
      
    

    """
       Returns the external configurations for the program
    """
    #user = args.user[0]
    global config
    if user.lower() == 'udi':
        config_file = "configUdi.cfg"
    elif user.lower() == 'rani':
        config_file = "configRani.cfg"
    elif user.lower() == 'shreyans':
        config_file = "configShreyans.cfg"
    elif user.lower() == "kaushal":
        config_file = "configUdi.cfg"
    if config is None:
        config = configparser.RawConfigParser() 
        my_path = os.path.abspath(os.path.dirname(__file__))
        my_path = my_path+"/"+config_file
        config.read(my_path)

    return config
    


