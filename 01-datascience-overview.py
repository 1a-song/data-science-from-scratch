#!/usr/bin/env python
# coding: utf-8

# In[2]:


#DEGREE CENTRALITY
#가장 중요한 사용자는 누구인가?

users = [
    {"id": 0, "name": "Hero"},
    {"id": 1, "name": "Dunn"},
    {"id": 2, "name": "Sue"},
    {"id": 3, "name": "Chi"},
    {"id": 4, "name": "Thor"},
    {"id": 5, "name": "Clive"},
    {"id": 6, "name": "Hicks"},
    {"id": 7, "name": "Devin"},
    {"id": 8, "name": "Kate"},
    {"id": 9, "name": "Klein"}
]

friendship_pairs = [(0,1), (0,2), (1,2), (1,3), (2,3), (3,4), (4,5), (5,6), (5,7), (6,8), (7,8), (8,9)]

friendships = {user['id']: [] for user in users}

for i, j in friendship_pairs:
    friendships[i].append(j)
    friendships[j].append(i)


# In[7]:


def number_of_friends(user):
    user_id = user['id']
    friend_ids = friendships[user_id]
    return len(friend_ids)

total_connections = sum(number_of_friends(user)
                       for user in users)

print(total_connections)


# In[8]:


num_users = len(users)
avg_connections = total_connections / num_users

print(avg_connections)


# In[11]:


num_friends_by_id = [(user['id'], number_of_friends(user))
                    for user in users]

num_friends_by_id.sort(key=lambda id_and_friends: id_and_friends[1],
                      reverse=True)

print(num_friends_by_id)


# In[14]:


#friends + friend of a friend
def foaf_ids_bad(user):
    return [foaf_id
           for friend_id in friendships[user['id']]
           for foaf_id in friendships[friend_id]]

foaf_ids_bad(users[0])


# In[20]:


#mutual friend가 몇명?
from collections import Counter

def friends_of_friends(user):
    user_id = user['id']
    return Counter(
        foaf_id
        for friend_id in friendships[user_id]
        for foaf_id in friendships[friend_id]
        if foaf_id != user_id
        and foaf_id not in friendships[user_id]
    )
    
print(friends_of_friends(users[3]))


# In[22]:


interests = [
    (0, "Hadoop"),(0, "Big Data"), (0, "HBase"), (0, "Java"), (0, "Spark"), (0, "Storm"), (0, "Cassandra"),
    (1, "NoSQL"), (1, "MongoDB"), (1, "Cassandra"), (1, "Postgres"), (1, "HBase"),
    (2, "numpy"), (2, "Python"), (2, "scikit-learn"), (2, "scipy"),
    (3, "statistics"), (3, "regression"), (3, "probability"),
    (4, "machine learning"), (4, "regression"), (4, "decision trees"), (4, "libsvm"),
    (5, "Haskell"), (5, "programming languages"), (5, "Java"), (5, "C++"), (5, "Python"), (5, "R"),
    (6, "probability"), (6, "mathematics"), (6, "theory"), (6, "statistics"),
    (7, "machine learning"), (7, "scikit-learn"), (7, "Mahout"), (7, "neural networks"), 
    (8, "neural networks"), (8, "artificial intelligence"), (8, "deep learning"), (8, "Big Data"), 
    (9, "Hadoop"), (9, "Java"), (9, "MpaReduce"), (9, "Big Data")
]

#특정 관심사를 가지고 있는 모든 사용자 id 반환
def data_scientists_who_like(target_interest):
    return [user_id
           for user_id, user_interest in interests
           if user_interest == target_interest]

from collections import defaultdict

#key = interest, value = id
user_ids_by_interest = defaultdict(list)

for user_id, interest in interests:
    user_ids_by_interest[interest].append(user_id)
    
#key = id, value = interest
interests_by_user_id = defaultdict(list)

for user_id, interest in interests:
    interests_by_user_id[user_id].append(interest)


# In[27]:


#추천 시스템
def most_common_interests_with(user):
    return Counter(
    interested_user_id
    for interest in interests_by_user_id[user['id']]
    for interested_user_id in user_ids_by_interest[interest]
    if interested_user_id != user['id']
    )

print(most_common_interests_with(users[0]))


# In[28]:


#자연어 처리
words_and_counts = Counter(word
                          for user, interest in interests
                          for word in interest.lower().split())

for word, count in words_and_counts.most_common():
    if count > 1:
        print(word, count)

