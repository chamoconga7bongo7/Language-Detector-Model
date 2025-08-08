import time
import joblib
import os
from datetime import datetime
import re
from nltk.tokenize import word_tokenize

start_time = datetime.now()
print("Started Time :",start_time)



class Test_on_model:

    def test_main(self):
        try:
            model = joblib.load("Model/model.pkl")
            while True:
                user_input = input("Enter Your Language here (type 'exist' to quit): ").strip()
                if user_input.lower() == "exist":
                    print("Exiting...")
                    break

                cleaned_input = re.sub(r"[^\w\s]", "", user_input)

                if not cleaned_input:
                    print("Input after cleaning is empty, please enter valid text.")
                    continue

                text_for_predict = " ".join(user_input)

                Predicted = model.predict([text_for_predict])[0]
                print("Predicted Language :", Predicted)

        except Exception as e:
            print(f"Failed to Test model : {e}")


C = Test_on_model()
C.test_main()



end_time = datetime.now()
print("End Time :",end_time)