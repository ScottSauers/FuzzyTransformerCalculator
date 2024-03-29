import random

def generate_number(digits):
    """Generate a number with the given number of digits."""
    if digits == 1:
        return random.randint(1, 9)
    return random.randint(10**(digits-1), (10**digits)-1)

def format_answer(answer):
    """Format the answer to avoid scientific notation and round to max 5 digits."""
    if "." in f"{answer}":
        return f"{round(answer, 5):.5f}".rstrip('0').rstrip('.')
    return str(answer)

def generate_problem(problem_type, digits):
    """Generate a single math problem, avoiding answers that result in zero."""
    num1 = generate_number(random.choice(digits))
    num2 = generate_number(random.choice(digits))
    if problem_type == 'division':
        # Skip division by zero or results in zero
        if num2 == 0 or num1 == 0:
            return None, None
        answer = num1 / num2
        if answer == 0:
            return None, None
        return f"{num1}/{num2}=", format_answer(answer)
    elif problem_type == 'addition':
        answer = num1 + num2
    elif problem_type == 'subtraction':
        answer = num1 - num2
    elif problem_type == 'multiplication':
        answer = num1 * num2

    if answer == 0:
        return None, None

    if not (0 <= answer <= 9):
        return None, None

    return f"{num1}{problem_type_symbol(problem_type)}{num2}=", format_answer(answer)

def problem_type_symbol(problem_type):
    """Return the symbol for the given problem type."""
    symbols = {
        'addition': '+',
        'subtraction': '-',
        'multiplication': '*',
        'division': '/'
    }
    return symbols.get(problem_type, '?')

def main():
    problems_count = 100000
    #problem_types = ['addition', 'subtraction', 'multiplication', 'division']
    problem_types = ['addition']
    #digits_distribution = [1, 1, 2, 2, 2, 3, 3, 4, 5]
    digits_distribution = [1, 1, 1, 1]
    problems_generated = 0

    with open('math_problems_dataset.txt', 'w') as file:
        while problems_generated < problems_count:
            problem_type = random.choice(problem_types)
            problem, answer = generate_problem(problem_type, digits_distribution)
            if problem is not None and answer is not None:
                file.write(f"{problem}{answer}\n")
                problems_generated += 1

if __name__ == "__main__":
    main()
    print("Done.")