import torch
import attention

STUDENT={'name': 'Eldar Hacohen',
         'ID': '311587661'}

def test_attention_scores():
    # fill in values for the a, b and expected_output tensor.
    a = torch.tensor([[[1, 2, 3, 4],
                       [2, 3, 4, 5]],
                      [[1, 2, 3, 4],
                       [2, 3, 4, 5]]])  # a three-dim tensor
    b = torch.tensor([[[1, 1, 1, 1],
                       [2, 2, 2, 2]],
                      [[2, 2, 2, 2],
                       [1, 1, 1, 1]]]) # a three-dim tensor
    # the expected_output should be a three-dim tensor of shape (1, 2, 2), normalized by sqrt(4)
    expected_output = torch.tensor([[[5, 10],
                                     [7, 14]],
                                    [[10, 5],
                                     [14, 7]]]).float()
    A = attention.attention_scores(a, b)

    # Note that we use "allclose" and not ==, so we are less sensitive to float inaccuracies.
    assert torch.allclose(A, expected_output)

if __name__ == '__main__':
    test_attention_scores()
    print("All tests passed!")