"""
Example file with type errors for Type-Guardian to fix
"""


def get_user(user_id):
    """Get user by ID"""
    users = {
        1: {"name": "Alice", "email": "alice@example.com"},
        2: {"name": "Bob", "email": "bob@example.com"},
    }
    return users.get(user_id)


def calculate_total(items, tax_rate):
    """Calculate total with tax"""
    subtotal = sum(item["price"] for item in items)
    tax = subtotal * tax_rate
    return subtotal + tax


def first_item(items):
    """Get first item from list"""
    if items:
        return items[0]
    return None


def process_users():
    """Process user list"""
    users = []
    
    for i in range(5):
        user = get_user(i)
        if user:
            users.append(user)
    
    return users


def main():
    """Main function"""
    user = get_user(1)
    email = user["email"]  # Potential None error
    print(f"Email: {email}")
    
    items = [
        {"price": 10.0},
        {"price": 20.0},
        {"price": 15.0},
    ]
    
    total = calculate_total(items, 0.08)
    print(f"Total: ${total:.2f}")
    
    users = process_users()
    first = first_item(users)
    print(f"First user: {first}")


if __name__ == "__main__":
    main()
