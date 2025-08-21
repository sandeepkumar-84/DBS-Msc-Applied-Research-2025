import os
from docling.document_converter import DocumentConverter
from docling_core.transforms.serializer.markdown import MarkdownDocSerializer

source  = "C:\Research-Chatbot\DataCollection\Brochures"
dest = "C:\Research-Chatbot\DataCollection\Brochures_text"
converter_files = DocumentConverter()

# Create destination directory if it doesn't exist
if not os.path.exists(dest):
    os.makedirs(dest)

for files in os.listdir(source):
    file_path = os.path.join(source, files)
    print(f"Processing file: {files}")
    try:
        doc_files = converter_files.convert(source=file_path)
        print(f"File Converted to doc: {files}")
        serializerMD = MarkdownDocSerializer(doc=doc_files.document)
        ser_result_md = serializerMD.serialize()
        print(f"File Serialized to doc: {files}")
        ser_text_md = ser_result_md.text
        # save at dest
        output_file_path = os.path.join(dest, files.replace('.pdf', '.txt'))
        with open(output_file_path, "w", encoding="utf-8") as file:
            file.write(ser_text_md)
        print(f"Successfully processed and saved: {files}")
    except Exception as e:
        print(f"Error processing file {files}: {e}")


print(f"Conversion completed successfully")