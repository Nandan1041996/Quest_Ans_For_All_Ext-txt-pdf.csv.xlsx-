document.addEventListener('DOMContentLoaded', function () {
    const askButton = document.getElementById('ask-question-btn');
    const resultBox = document.getElementById('result-box');
    const loaderContainer = document.getElementById('loader-container'); // Ensure loader is correctly referenced
    const selectedFile = document.getElementById('selected_file');
    const selectedLanguage = document.getElementById('selected_language');
    const queryText = document.getElementById('query_text');
    const docSelectionMessage = document.getElementById('doc-selection-message');

    function updateButtonVisibility() {
        askButton.style.display = (selectedFile.value !== '' && selectedLanguage.value !== '') ? 'block' : 'none';
    }

    // Function to handle sending data to answers.json
    function sendToAnswersJson(prompt, chatAns, humanAns) {
        const answerData = {
            Prompt: prompt,
            "Chat-Ans": chatAns,
            "Human-Ans": humanAns
        };

        fetch('/save_answers', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(answerData)
        })
        .then(response => response.json())
        .then(data => {
            console.log('Answer data saved:', data);
        })
        .catch(error => {
            console.error('Error saving answer data:', error);
        });
    }

    askButton.addEventListener('click', function () {
        if (selectedFile.value === '') {
            docSelectionMessage.textContent = 'Please select a document.';
            return;
        } else {
            docSelectionMessage.textContent = ''; // Clear message if valid
        }

        // Show loader and hide ask button
        loaderContainer.style.display = 'flex'; // Show loader as flex to center it
        askButton.style.display = 'none'; // Hide ask button

        const formData = new FormData();
        formData.append('query_text', queryText.value);
        formData.append('selected_file', selectedFile.value);
        formData.append('selected_language', selectedLanguage.value);

        fetch('/ask', {
            method: 'POST',
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            const question = queryText.value;
            const answer = data.answer;

            // Show question in chatbox
            resultBox.innerHTML += `
            <div class="message-block">
                <div class="message user">${question}</div>
                <div class="message bot">${answer}</div>
                <div class="feedback-section">
                    <div class="feedback-icons">
                        <span class="feedback correct">&#x2714;</span> <!-- Right symbol -->
                        <span class="feedback wrong">&#x274C;</span> <!-- Wrong symbol -->
                    </div>
                    <div class="feedback-box" style="display: none;">
                        <div class="feedback-input">
                            <textarea class="feedback-text" placeholder="Please provide your feedback..."></textarea>
                            <button class="submit-feedback">Submit</button>
                        </div>
                        <span class="feedback-error" style="color:red; display:none;">Please provide feedback.</span>
                    </div>
                </div>
            </div>`;
        
        

            // Scroll to the bottom of the chatbox
            resultBox.scrollTop = resultBox.scrollHeight;

            // Send answer data to answers.json if user clicks correct or wrong
            document.querySelectorAll('.feedback.correct').forEach((correctButton, index) => {
                correctButton.addEventListener('click', function () {
                    // Save correct answer in JSON
                    const correctQuestion = document.querySelectorAll('.message.user')[index].textContent;
                    const correctAnswer = document.querySelectorAll('.message.bot')[index].textContent;

                    sendToAnswersJson(correctQuestion, correctAnswer, ""); // Human answer is empty for correct
                    saveFeedback(correctQuestion, correctAnswer, true);
                });
            });

            document.querySelectorAll('.feedback.wrong').forEach((wrongButton, index) => {
                wrongButton.addEventListener('click', function () {
                    // Show feedback box for wrong answers
                    const feedbackBox = document.querySelectorAll('.feedback-box')[index];
                    feedbackBox.style.display = 'block';

                    // Handle submit feedback
                    const submitFeedbackButton = feedbackBox.querySelector('.submit-feedback');
                    submitFeedbackButton.addEventListener('click', function () {
                        const feedbackText = feedbackBox.querySelector('.feedback-text').value;
                        const feedbackError = feedbackBox.querySelector('.feedback-error');

                        // Check if feedback is empty
                        if (feedbackText.trim() === '') {
                            feedbackError.style.display = 'block'; // Show error message
                        } else {
                            feedbackError.style.display = 'none'; // Hide error message

                            const wrongQuestion = document.querySelectorAll('.message.user')[index].textContent;
                            const wrongAnswer = document.querySelectorAll('.message.bot')[index].textContent;

                            sendToAnswersJson(wrongQuestion, wrongAnswer, feedbackText); // Include feedback
                            saveFeedback(wrongQuestion, wrongAnswer, false, feedbackText);

                            // Hide the feedback box after submission
                            feedbackBox.style.display = 'none';
                        }
                    });
                });
            });

            // Clear the input text area after sending the question
            queryText.value = ''; 
        })
        .catch(error => {
            console.error('Error:', error);
        })
        .finally(() => {
            loaderContainer.style.display = 'none'; // Hide loader
            askButton.style.display = 'block'; // Show ask button again
        });
    });

    // Handle Enter and Shift+Enter in the textarea
    queryText.addEventListener('keypress', function (event) {
        if (event.key === 'Enter') {
            if (event.shiftKey) {
                // Allow new line
                return;
            } else {
                // Trigger ask button click
                askButton.click();
                event.preventDefault(); // Prevent new line
            }
        }
    });

    selectedFile.addEventListener('change', function () {
        if (selectedFile.value === '') {
            docSelectionMessage.textContent = 'Please select a document.';
        } else {
            docSelectionMessage.textContent = '';
        }
        updateButtonVisibility();
    });

    selectedLanguage.addEventListener('change', function () {
        updateButtonVisibility();
    });

    // Initialize button visibility on page load
    updateButtonVisibility();

    // Function to save feedback in a JSON file
    function saveFeedback(question, answer, isCorrect, feedback = '') {
        const feedbackData = {
            question: question,
            answer: answer,
            isCorrect: isCorrect,
            feedback: feedback
        };

        fetch('/save_feedback', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(feedbackData)
        })
        .then(response => response.json())
        .then(data => {
            console.log('Feedback saved:', data);
        })
        .catch(error => {
            console.error('Error saving feedback:', error);
        });
    }
});