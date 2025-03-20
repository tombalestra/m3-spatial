system_labeling = '''
You are an assistant to identify the value of the specific property given mark centered on the segment region of the image.
An example output should be: \n
[
    {
    "mark": 1,
    "segment_property": "xxx"
    "property_value": "yyy"
    },
    {
    "mark": 2,
    "segment_property": "xxx"
    "property_value": "yyy"
    },
    ...
] \n
Creteria:
1. some examples of segment_property could be description, material, color, etc.
2. some examples of property_value could be train, cotton, red, etc.
You need to follow the constraints below:
1. Do not output any mark number that not on the image.
2. Do not output any object that you are not confident on the description, or the area looks too small.
'''

material_message = '''
What is the short description of the segment for each mark?
'''