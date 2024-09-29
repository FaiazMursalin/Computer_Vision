import torch

'''Question 1: Create a Tensor with Constraints
Task: Write a PyTorch script to create a tensor of size 4x4 filled with random integers from 0 to 10. 
Ensure that no value in the tensor exceeds 8.
Hint: You may need to generate the tensor and then modify it to fit the constraint.'''

# tensor with 4x4 with random number from inclusive 0 to exclusive 11 that mean from 0-10
tensor_1 = torch.randint(0, 11, (4, 4))
# ensuring no value exceeds 8
tensor_1 = torch.clamp(tensor_1, max=8)
print("Tensor for task 1 :", tensor_1)

'''Question 2: Modify Tensor Using Conditionals
Task: For the tensor created in Question 1, replace all values greater than 5 with the value 5.
Hint: Utilize conditional indexing to manipulate tensor values.'''
# cloning the tensor1 from previous task
tensor_2 = tensor_1.clone()
# conditional indexing
tensor_2[tensor_2 > 5] = 5
print("Tensor for task 2 :", tensor_2)

'''Question 3: Matrix Multiplication
Task: Perform matrix multiplication between the modified tensor from Question 2 and a new 
4x4 tensor filled with random integers between 1 and 3.
Hint: Use torch.matmul or the @ operator for matrix multiplication.'''
# new tensor values within 1-3 size 4x4
tensor_3 = torch.randint(1, 4, (4, 4))
# matrix multiplication with tensor from task 2 with the tensor created above
result_tensor_task_3 = torch.matmul(tensor_2, tensor_3)
print("Tensor for task 3 :", result_tensor_task_3)

'''Question 4: Calculate the Sum and Standard Deviation
Task: From the resulting tensor of Question 3, compute the sum of all elements and the standard deviation.
Hint: Use the .sum() and .std() tensor methods.'''
sum_result_task_4 = torch.sum(result_tensor_task_3)
print("The sum of the tensor from previous tensor in question 3: ", sum_result_task_4.item())
# converted to float as i used randint so
std_result_task_4 = torch.std(result_tensor_task_3.float())
print("The standard deviation of the previous tensor in question 3: ", std_result_task_4.item())

'''Question 5: Concatenate and Reshape Tensors
Task: Concatenate the tensor from Question 1 with the tensor from Question 3 along a new dimension, 
and then reshape the result into a tensor of shape 2x16.
Hint: Use torch.cat() to concatenate tensors and .view() to change the shape.'''
concatenated_tensor = torch.cat((tensor_1, result_tensor_task_3), dim=0)
reshaped_concatenated_tensor = concatenated_tensor.view(2, 16)
print("reshaped_concatenated_tensor: ", reshaped_concatenated_tensor)
