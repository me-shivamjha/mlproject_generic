import sys   # sys module provides access to some variables used or maintained 
            #by the interpreter and to functions that interact strongly with the interpreter.

def error_message_detail(error, error_detail: sys):
    _, _, exc_tb = error_detail.exc_info()  # exc_info() returns a tuple containing information about the exception that is currently being handled.
    
    
    file_name = exc_tb.tb_frame.f_code.co_filename  # tb_frame is a reference to the stack frame where the exception occurred. 
                                                    # f_code is a reference to the code object being executed in that frame, and co_filename is the name of the file from which the code was loaded.
    line_number = exc_tb.tb_lineno  # tb_lineno is an attribute of the traceback object that indicates the line number in the source code where the exception occurred.
    error_message = f"Error occurred in script: {file_name} at line number: {line_number} with error message: {str(error)}"
    return error_message

class CustomException(Exception):
    def __init__(self, error_message, error_detail: sys):
        super().__init__(error_message)  # Call the constructor of the base class (Exception) to initialize the error message.
        self.error_message = error_message_detail(error_message, error_detail)  # Generate a detailed error message using the error_message_detail function.

    def __str__(self):
        return self.error_message  # Return the detailed error message when the exception is converted to a string.