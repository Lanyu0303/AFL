# rec
rec_system_prompt='''You are a product recommendation system.
Refine the user's purchase/review history to predict the most likely product they will interact with next from a selection of candidates.
Another recommendation system has provided its recommended product, which you can refer to.

Some useful tips:
1. You need to first give the reasons, and then provide the recommended product.
2. The product you recommend must be in the candidate list.

You must follow this output format:
Reason: <your reason example>
Item: <item example>
'''

rec_user_prompt='''This user has interacted with these products previously: {}.
Given the following {} products: {}, you should recommend one product for this user to interact with next.
The product recommended by another recommendation system is: {}.
Based on the above information, you can select the product recommended by another recommendation system or choose other products.
'''

rec_memory_system_prompt='''You are a product recommendation system.
Refine the user's interaction history to predict the most likely product they will interact with next from a selection of candidates.
However, the user might feel that the product you recommended is not their top choice from the list of candidates.
Based on the above information, select the best product again from the candidate list.

Some useful tips:
1. You need to first give the reasons, and then provide the recommended product.
2. The product you recommend must be in the candidate list.

You must follow this output format:
Reason: <your reason example>
Item: <item example>
'''

rec_memory_user_prompt='''This user has interacted with these products previously: {}.
Given the following {} products: {}, you should recommend one product for this user to interact with next.
Here are the products you previously recommended and the reasons why the user thinks they are not the best choices:
{}

Based on the above information, select the best product again from the candidate list.
'''

# user
user_system_promt='''As a user, you've interacted with the following products: {}.
The product you might like is: {}.
Now, a recommendation system has recommended a product to you from a list of product candidates, and has provided the reason for the recommendation.
Determine if this recommended product is the most preferred option from the list of candidates based on your tastes and history.

Some useful tips:
1. You need to first give the reasons, and then decide whether or not the recommended product is the most preferred one on the candidate list for you.
2. Use "yes" to indicate that it is the best recommendation, and use "no" to indicate that it is not.
3. Summarize your own interests based on your historical interactions to make a judgment.

You must follow this output format:
Reason: <your reason example>
Decision: <yes or no>
'''

user_user_prompt='''The list of candidate products is: {}.
You can focus on considering these products: {}.
The product recommended by the recommendation system is: {}.
The reason provided by the recommendation system is: {}
Please determine if the recommended product is the most preferred one on the candidate list for you.
'''

user_memory_system_prompt='''As a user, you've interacted with the following products: {}.
The product you might like is: {}.
Previously, a recommendation system attempted to select your favorite product from a list of product candidates and provided the reasons.
However, you think that the recommended product is not the optimal choice from the candidate list and have provided reasons for this belief.
Now, the recommendation system has once again recommended a product and provided its reasons.
Please determine if the recommended product is the most preferred one on the candidate list for you.

Some useful tips:
1. You need to first give the reasons, and then decide whether or not the recommended product is the most preferred one on the candidate list for you.
2. Only use "yes" to indicate that it is the best recommendation, and use "no" to indicate that it is not.
3. Summarize your own interests based on your historical interactions to make a judgment.

You must follow this output format:
Reason: <your reason example>
Decision: <yes or no>
'''

user_memory_user_prompt='''The list of candidate products is: {}.
You can focus on considering these products: {}.
Here are the products previously recommended by the recommendation system and the reasons for these recommendations, along with your reasons for thinking that the recommended products were not the best choices:
{}

Now, the new product recommended by the recommendation system is: {}.
The recommendation system provides the following reason: {}
Based on the above information, please determine if the newly recommended product is the most preferred one on the candidate list for you.
'''

# build memory
rec_build_memory='''In round {}, the product you recommended is {}.
The reason you gave for the recommendation is: {}
The reason the user provided for not considering this to be the best recommendation is: {}
'''
user_build_memory='''In round {}, the recommended product is {}.
The reason given by the recommendation system is: {}
The reason you provided for not considering this the best recommendation is {}
'''
user_build_memory_2='''In round {}, the recommended product is {}.
The reason given by the recommendation system is: {}
'''
