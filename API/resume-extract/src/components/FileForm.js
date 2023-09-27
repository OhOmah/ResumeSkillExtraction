import {useState} from 'react';

function FileForm() {
    const [file, setFile] = useState(null);
    
    const handleFileInputChange = (event) => {
        console.log(event.target)
        setFile(event.target.files[0])
    }

    const handleSubmit = async (event) => {
        event.preventDefault();

        const formData = new FormData();
        formData.append('file_upload', file)

        try {
            const endpoint = "http://localhost:8000/uploadfile/"
            const response = await fetch(endpoint, {
                method: "POST",
                body: formData
            });

            if (response.ok) {
                console.log("file uploaded successfully!")
            } else {
                console.error("Failed to upload file.");
            }
        } catch(error) {
            console.error(error);
        }
    }

    return (
        <div>
            <h1>Upload File</h1>

            <form onSubmit={handleSubmit}>
                <div style= {{ marginBottom: "20px"}}>
                <input type="file" onChange={handleFileInputChange} />
                </div>

                <button type="submit">Upload</button>
            </form>

            { file && <p>{file.name}</p>}
        </div>
    )
}

export default FileForm