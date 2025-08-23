cart=["snacks","chocolate","veggies","biscuits","fruits"]
print(cart)
cart.append("drinks")
print(cart)
cart.remove("chocolate")
print(cart)
cart[1]="candy"
print(cart)
for index,item in enumerate(cart):
    print(index,item)
if"candy"in cart:
    print("candy is found",cart.index("candy"))
    print(len(cart))
    print(cart[1:3])
    print(cart[2:4])
    cart.sort()
    print(cart)
    print("total length of my cart is:",len(cart),cart)