#!/usr/bin/env python3
"""Module for answer question"""
tf_question_answer = __import__('0-qa').tf_question_answer


def answer_loop(reference):
    """Function that answers questions from a reference text
        reference: reference text"""
    # Keep rolling until the user exits
    while True:
        # Recieve user input with Q:
        user_words = input("Q: ")

        # Check to see if exit command is used
        if user_words.lower() in ["bye", "exit", "goodbye", "quit"]:
            print("A: Goodbye")
            break

        # Get answer using tf_question_answer function
        answer = tf_question_answer(user_words, reference)

        # if there is not an answer, respond with default message
        if answer is None:
            print("A: Sorry, I do not understand your question.")
        else:
            print(f"A: {answer}")
