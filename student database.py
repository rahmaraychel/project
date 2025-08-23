#student_database={"name":"","roll":""}
#dict1={}
#num_entries=int()

my_dict = {}
num_entries = int(input("Enter the number of key-value pairs you want to add: "))

for i in range(num_entries):
    key = input(f"Enter key: ")
    value = input(f"Enter value: ")
    my_dict[key] = value
    print("updated dictionarr:",my_dict)

a=input(f"enter the value to delete:")
del my_dict[a]
print(my_dict)
b=input(f"enter the value to search:")
search_key="b"
if search_key in my_dict:
    print(f"'{search_key}'is present in dictionary.")
else:
    print(f"'{search_key}'is not present in dictionary.")    
 

    



