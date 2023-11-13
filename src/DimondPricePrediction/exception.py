import sys

# creating a class for custom exception
class customexception (Exception):

    # creating 2 parameters namely error message and error details while initializing the constructor
    def __init__(self, error_message, error_details:sys):

        self.error_message=error_message

        _,_,exc_tb=error_details.exc_info()
        # we can also write as - exception_type, exception_class, exc_tb = error_details.exc_info()
        # here exception_type and exception_class is not as such important. that's why we have placed them in placeholders _.
        # also exc_tb (full form exeption traceback) gives information regarding line no. and file name where the error occured.

        self.line_no = exc_tb.tb_lineno     # for extracting info about error occurance line no. stored in exc_tb
        self.file_name = exc_tb.tb_frame.f_code.co_filename    # for extracting name of module where error occured

    # for the format of the output message in the terminal about the error
    def __str__(self):
        return "Error occured in python script name [{0}] at line number [{1}] with error message as [{2}]".format(self.file_name, self.line_no, str(self.error_message))



# testing the above code
if __name__ == "__main__":
    
    try:
        a=5
        b=0
        print(a/b)

    except Exception as e:
        raise customexception(e, sys)