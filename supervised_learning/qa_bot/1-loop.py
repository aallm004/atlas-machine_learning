#!/usr/bin/env python3
"""Script that takes in input from the usesr with the prompt 'Q:' and prints
'A:' as a response. If the user inputs 'exit', 'quit', 'goodbye', or 'bye',
case insensitive, print 'A: Goodbye' and exit"""


while True:
    user_input = input("Q: ")

    # Check if the user wants to exit
    if user_input.lower() in ["exit", "quit", "goodbye", "bye"]:
        print("A: Goodbye")
        break
    else:
        # For any other input, just print "A:" for now
        print("A:")
