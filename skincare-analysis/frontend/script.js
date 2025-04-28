async function uploadImage() {
  const input = document.getElementById('fileInput');
  const file = input.files[0];
  const formData = new FormData();
  formData.append('image', file);

  const response = await fetch('http://localhost:5000/predict', {
      method: 'POST',
      body: formData
  });

  const data = await response.json();
  const outputDiv = document.getElementById('output');
  outputDiv.innerHTML = `<img src="http://localhost:5000${data.output_image}" />`;
}