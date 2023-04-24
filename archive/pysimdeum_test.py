from pysimdeum import pysimdeum
import matplotlib.pyplot as plt

for i in range(5):
    house = pysimdeum.built_house(house_type='family', user_num=4)
    if i % 2 == 0:
        house.users[0].age = 'home_ad'
        house.users[0].job = False

    for user in house.users:
        user.age = 'work_ad'
        user.job = True

    print(house.users)

    consumption = house.simulate(num_patterns=100)
    tot_cons = consumption.sum(['enduse', 'user']).mean([ 'patterns'])
    tot_cons.groupby('time.hour').mean().plot()
plt.legend(labels=range(5))
plt.savefig('pysimdeum_plot')
