import math

EOS = 0

if __name__ == "__main__":
    prompt = "Hello world"
    model = lambda x: x

    input = prompt
    response = None
    max_output_length = math.inf()

    for i in range(max_output_length):
        output = model(input)

        if output == EOS:
            break
    
        input += output
        response += output
    
    print(response)
