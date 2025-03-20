system_msg = '''
You are an assistant to generate 
(1) A long grounding description for each mark in Image1. 
(2) A short grounding description for the mark for large and confidence regions in Image1. (e.g. Train, Sky, Car, etc.)
(3) A long Language description for Image1, generate the caption that has distinguishable contents for Image1 in compared with Image2, and Image3.
(4) A short Language description for Image1, generate the caption that has distinguishable contents for Image1 in compared with Image2, and Image3.
Constraints:
A. Don't generate grouding description for the mark you don't see in Image 1.
B. Image1, Image2, Image3 are labeled on the top right corner of the given images.
C. Generate the data format that is able to be loaded by json.loads().
D. Do not start with Image1 xxx, just make the caption as a whole.
E. For short grounding description, only consider the object that is large and confident.
'''

user_msg = '''
Given input Image1, Image2, Image3, please generate (1), (2) for Image1 in following format:
Output:
{
    "Long Grounding": {
        "1": "description xxx.",
        "2": "description xxx.",
        ...
    },
    "Short Grounding": {
        "1": "description xxx.",
        "2": "description xxx.",
        ...
    },
    "Caption Long": "description xxx."
    "Caption Short": "description xxx."
}
'''

system_msg2 = '''
You are an assistant to generate a description for the mark in Image1. Here are the requirements:
(2) The description should be short and concise.
(3) Please only focus on the main objects in the image.
(4) The mark is in the center of the object.
Constraints:
A. Don't generate grounding description for the mark you don't see in Image.
B. Don't generate grounding description that you are not confident about.
'''

system_msg_garden = '''
You are an assistant to generate grounding description for image1, here are the requirements:
(1) You have the freedom longer description than the proposal names.
(2) The image contains [door, tree, tree trunk, plant pot, watering can, bushes, bush with flower, brick wall, grassland, table, stone paving], please generate the grounding description according the candidate objects or beyond.
(3) Please only focus on the main objects in the image, if the mark is overlap/ambiguous, please don't generate the description.
(4) The mark is in the center of the object.
Constraints:
A. Only generate max 8 most confident grounding description for the marks in Image1.
B. Don't generate grounding description for the mark you don't see in Image.
C. Don't generate grounding description that you are not confident about.
D. There are color overlay on the image, please ignore the overlayed color.
'''

system_msg_train = '''
You are an assistant to generate grounding description for image1, here are the requirements:
(1) You have the freedom longer description than the proposal names.
(2) The image contains [sky, train, tree, mountain, wall, car, truck, cone, gravel, railroad], please generate the grounding description according the candidate objects or beyond.
(3) Please only focus on the main objects in the image, if the mark is overlap/ambiguous, please don't generate the description.
(4) The mark is in the center of the object.
Constraints:
A. Only generate max 5 most confident grounding description for the marks in Image1.
B. Don't generate grounding description for the mark you don't see in Image.
C. Don't generate grounding description that you are not confident about.
D. There are color overlay on the image, please ignore the overlayed color.
'''

system_msg_tabletop = '''
You are an assistant to generate grounding description for image1, here are the requirements:
(1) You have the freedom longer description than the proposal names.
(2) The image contains [yellow duck, red ball, green plush toy, orange scissors, orange sponge, role of tape, small blue bottle, white water bottle, mustard bottle, blue spiked ball, water melon, strawberry], please generate the grounding description according the candidate objects or beyond.
(3) Please only focus on the main objects in the image, if the mark is overlap/ambiguous, please don't generate the description.
(4) The mark is in the center of the object.
Constraints:
A. Only generate max 5 most confident grounding description for the marks in Image1.
B. Don't generate grounding description for the mark you don't see in Image.
C. Don't generate grounding description that you are not confident about.
D. There are color overlay on the image, please ignore the overlayed color.
'''

system_msg_drjohnson = '''
You are an assistant to generate grounding short and simple description for image1, here are the requirements:
(1) You have the freedom longer description than the proposal names.
(2) The image contains [green wall, large picture frame, small picture frame, chair under the big frame, white shuttered windows, white wall, white door, fire extinguisher], please generate the grounding description according the candidate objects or beyond.
(3) Please only focus on the main objects in the image, if the mark is overlap/ambiguous, please don't generate the description.
(4) The mark is in the center of the object.
Constraints:
A. Only generate max 5 most confident grounding description for the marks in Image1.
B. Don't generate grounding description for the mark you don't see in Image.
C. Don't generate grounding description that you are not confident about.
D. There are color overlay on the image, please ignore the overlayed color.
'''

system_msg_geisel = '''
You are an assistant to generate grounding short and simple description for image1, here are the requirements:
(1) You have the freedom longer description than the proposal names.
(2) The image contains [library architecture, sky, trees, grasses, mountain, land], please generate the grounding description according the candidate objects or beyond.
(3) Please only focus on the main objects in the image, if the mark is overlap/ambiguous, please don't generate the description.
(4) The mark is in the center of the object.
Constraints:
A. Only generate max 5 most confident grounding description for the marks in Image1.
B. Don't generate grounding description for the mark you don't see in Image.
C. Don't generate grounding description that you are not confident about.
D. There are color overlay on the image, please ignore the overlayed color.
'''

system_msg_playroom = '''
You are an assistant to generate grounding short and simple description for image1, here are the requirements:
(1) You have the freedom longer description than the proposal names.
(2) The image contains [ceiling, monitor, table, bookshelf, wall, door, books], please generate the grounding description according the candidate objects or beyond.
(3) Please only focus on the main objects in the image, if the mark is overlap/ambiguous, please don't generate the description.
(4) The mark is in the center of the object.
Constraints:
A. Only generate max 5 most confident grounding description for the marks in Image1.
B. Don't generate grounding description for the mark you don't see in Image.
C. Don't generate grounding description that you are not confident about.
D. There are color overlay on the image, please ignore the overlayed color.
'''


user_msg2 = '''
Given input Image1, please generate (1), (2) for Image1 in following format:
{
    "Grounding": {
        "1": "description xxx.",
        "2": "description xxx.",
        ...
    },
}
'''

user_msg3 = '''
Given input Image1, please generate the image caption in following format:
{
    "Short Caption": "description xxx.",
    "Middle Caption": "description xxx.",
    "Long Caption": "description xxx.",
}
'''

system_msg3 = '''
You are an assistant to generate caption for image1, here are the requirements:
(1) You are required to generate a short and concise caption, a middle length caption to describe the scene, and a long and detailed caption.
Constraints:
A. There are color overlay on the image, please ignore the overlayed color.
'''
