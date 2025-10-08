# SONI'S SHOP OF SOME ITEMS............


list=['kurkure','chips','candy','cake','icecream']
thing=input('enter the thing that you want to eat: ')
if thing in list:
    if thing==list[0]:
        print('take it and give only 10 rs')
    
    if thing==list[1]:
        print('take it and give only 20 rs')
    
    if thing == list[2]:
        print('take it and give only 5 rs')
    
    if thing==list[3]:
        print('which flavor do you want to have?')
        cake_flavor_with_price={
            'vanilla': 150,
            'choclate': 160,
            'strawberry': 170,
        }
        flavor=input('enter the flavor: ')
        if flavor in cake_flavor_with_price:
            print('your cake is ready with price of',cake_flavor_with_price[flavor])
        else:
            print('sorry we dont have this flavor')
    if thing == list[4]:
        print('which flavor do you want to have?')
        icecream_flavor_with_price={
            'vanilla': 50,
            'choclate': 60,
            'strawberry': 70,
        }
        flavor=input('enter the flavor: ')
        if flavor in icecream_flavor_with_price:
            print('your icecream is ready with price of',icecream_flavor_with_price[flavor])
        else:
            print('sorry we dont have this flavor')
else:
    print('sorry,we do not have this item')
