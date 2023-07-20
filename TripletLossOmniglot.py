import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import Omniglot
from torchvision.transforms import transforms
import random


class SiameseNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(11025, 4096)
        self.fc2 = nn.Linear(4096, 1024)
        self.fc3 = nn.Linear(1024, 128)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x


############## GETS TRIPLETS DATASET ##############
class TripletsDataset(Dataset):
    """
    The triplets dataset will be in the form (18316 triplets in total) (105 pixels x 105 pixels = 11025):
    [
        [
            [x, x, ...] ANCHOR 1 x 11025
            [x, x, ...] POSITIVE
            [x, x, ...] NEGATIVE
        ],
        [
            [x, x, ...] ANCHOR
            [x, x, ...] POSITIVE
            [x, x, ...] NEGATIVE
        ],
        ...
    ]
    """
    def __init__(self):
        # omni_train has 20 examples of 1st char, then 20 examples of 2nd char, etc... has 964 chars in total (20 examples of each)
        omni_train = Omniglot(download=True, root='./', transform=transforms.ToTensor(), background=True)

        # Triplets are of the form (A, P, N) Anchor Positive Negative
        # There will 18316 triplets in total (19 * 964 (as one example serves as anchor))
        self.triplets = torch.zeros([18316, 3, 11025], dtype=torch.float32)

        for i in range(964):
            # Numbers representing other 963 characters (current character is removed)
            other_chars_index = [k for k in range(964)]
            other_chars_index.pop(i)

            for j in range(1, 20):
                # Gets anchor   0, 20, 40, ... (20 * 964)  (same for every iteration of j for loop)
                anchor = omni_train[i * 20][0]

                # Gets positive   anchor_index + 1, + 2, + 3, ... + 19
                positive = omni_train[i * 20 + j][0]

                # Gets negative   index of start of examples of different char, then adds 0 - 19
                other_char_index = other_chars_index[random.randint(0, 962)]
                negative = omni_train[other_char_index * 20 + random.randint(0, 19)][0]

                anchor = anchor.view(1, 11025)
                positive = positive.view(1, 11025)
                negative = negative.view(1, 11025)

                self.triplets[i * 19 + j - 1][0] = anchor
                self.triplets[i * 19 + j - 1][1] = positive
                self.triplets[i * 19 + j - 1][2] = negative

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, item):
        return self.triplets[item]


############## GETS SUPPORT SET ##############
# Even though omni_test has 659 different chars, an alphabet of only 20 chars will be used, to make testing quicker.
omni_test = Omniglot(download=True, root='./', transform=transforms.ToTensor(), background=False)

"""
Support set will be in the form (20 characters in the form 1 x 11025):
[
    [x, x, ...], 1 x 11025
    [x, x, ...],
    ...
]
"""
support_set = torch.zeros([20, 11025], dtype=torch.float32)

for i in range(20):
    support_set[i] = omni_test[i * 20][0].view(1, 11025)


############## GETS TEST DATASET ##############
class TestDataset(Dataset):
    def __init__(self):
        """
        The test dataset will be in the form (380 image answer pairs, as there are 19 examples for each char in the 20
        character test alphabet (as one has been used in support_set)):
        [
            [[x, x, ...], y],
            [[x, x, ...], y],
            ...
        ]
        (NOTE: y-values correspond to INDEX in support_set)
        """
        self.test_dataset = []
        for k in range(20):
            for m in range(19):
                image_answer_pair = []

                # Plus 1 due to the first example being used for support_set, e.g. 1, 2, 3, ..., 19
                image = omni_test[k * 20 + 1 + m][0].view(11025)
                answer = omni_test[k * 20 + 1 + m][1]

                image_answer_pair.append(image)
                image_answer_pair.append(answer)

                self.test_dataset.append(image_answer_pair)

    def __len__(self):
        return len(self.test_dataset)

    def __getitem__(self, item):
        return self.test_dataset[item]


############## TRAINING AND TESTING ##############
triplets = TripletsDataset()
test_dataset = TestDataset()

triplets_loader = DataLoader(triplets, shuffle=True, batch_size=1)
test_loader = DataLoader(test_dataset)

siamese_net = SiameseNet()
optimizer = optim.Adam(siamese_net.parameters(), lr=0.000001)
EPOCHS = 50
ALPHA = 1

for epoch in range(EPOCHS):
    print(f'\nEpoch {epoch + 1}:')
    for i, triplet in enumerate(triplets_loader):
        optimizer.zero_grad()

        anchor = siamese_net(triplet[0][0])
        positive = siamese_net(triplet[0][1])
        negative = siamese_net(triplet[0][2])

        difference_anchor_positive = anchor - positive
        difference_anchor_negative = anchor - negative

        vector_norm_dap = torch.norm(difference_anchor_positive)
        vector_norm_dan = torch.norm(difference_anchor_negative)

        # Triplet loss max(||f(a) - f(p)||^2 - ||f(a) - f(n)||^2 + alpha, 0)
        # Optimization works the same by just getting the first part, and only updating if > 0
        loss = torch.square(vector_norm_dap) - torch.square(vector_norm_dan) + ALPHA

        if loss > 0:
            loss.backward()
            optimizer.step()

        # 18315 will be index at 100% of training
        if (i % 100 == 0 and i != 0) or i == 18315:
            correct = 0
            total = 0

            with torch.no_grad():
                # Outputs of each support_set char
                support_set_outputs = torch.zeros([20, 128], dtype=torch.float32)

                for k in range(20):
                    support_set_outputs[k] = siamese_net(support_set[k])

                for m, (X, y) in enumerate(test_loader):
                    test_output = siamese_net(X[0])

                    difference_scores = torch.zeros(20, dtype=torch.float32)

                    for j in range(20):
                        difference_scores[j] = torch.square(torch.norm(support_set_outputs[j] - test_output))

                    # Least different support_set example is answer
                    if torch.argmin(difference_scores) == int(y):
                        correct += 1
                    total += 1

            print(f'{correct / total * 100:.1F}% accuracy at {(i + 1) / 18316 * 100:.1F}% of training')













