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

    const handleJobTitles = async (event) => {
        event.preventDefault();

        try {
            const endpoint = "http://localhost:8000/pulljobtitles"
            const response = await fetch(endpoint, {
                method: "POST",
                body: "test"
            });

            if (response.ok) {
                console.log("Job Titles Pulled Successfully!")
            } else {
                console.error("Failed to run model.");
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
            <div>
                <h1>Pull Job Titles</h1>

                <form onSubmit={handleJobTitles}>
                    <div style= {{ marginBottom: "20px"}}>
                    </div>
                    <button type="submit">Pull Job Titles</button>
                </form>
            </div>
        </div>
    )
}

export default FileForm