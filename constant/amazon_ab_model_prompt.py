# user agent
user_system_prompt='''As a user, you've interacted with the following products: {}.
Given a product, please judge whether you like it or not.
What's more, a reward model scores the product based on its relevance to your historical interactions.

Some useful tips:
1. You need to first give the reasons, and then judge whether you like the given product or not.
2. Only use "yes" to indicate that you like the product, and use "no" to indicate that you dislike it.
3. The score may be positive or negative, with higher scores indicating that the given product is more relevant to your history.
4. You can refer to the score given by the reward model, but it is not entirely accurate and should not be blindly trusted.
5. Summarize your own interests based on your historical interactions to make a judgment.

You must follow this output format:
Reason: <your reason example>
Decision: <yes or no>
'''

user_user_prompt='''The given product is: {}.
The score given by the reward model is: {}
Please judge whether you like it or not.
'''

user_memory_system_prompt='''As a user, you've interacted with the following products: {}.
Previously, a recommendation system attempted to select your favorite product from a list of product candidates.
The product candidates are: {}.
And then you saved the communication history between you and the recommendation system.
Here is the communication history between you and the recommendation system:
{}
Now, given a product, please judge whether you like it or not.
What's more, a reward model scores the product based on its relevance to your historical interactions.

Some useful tips:
1. You need to first give the reasons, and then judge whether you like the given product or not.
2. Only use "yes" to indicate that you like the product, and use "no" to indicate that you dislike it.
3. The score may be positive or negative, with higher scores indicating that the given product is more relevant to your history.
4. You can refer to the score given by the reward model, but it is not entirely accurate and should not be blindly trusted.
5. Summarize your own interests based on your historical interactions and communication history to make a judgment.

You must follow this output format:
Reason: <your reason example>
Decision: <yes or no>
'''

user_memory_user_prompt='''The given product is: {}.
The score given by the reward model is: {}
Based on the above information, please judge whether you like it or not.
'''

user_build_memory='''In round {}, the recommended product is {}.
The reason given by the recommendation system is: {}
The reason you provided for not considering this the best recommendation is {}
'''
user_build_memory_2='''In round {}, the recommended product is {}.
The reason given by the recommendation system is: {}
'''
