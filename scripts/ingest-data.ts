import fs from 'fs';
import { RecursiveCharacterTextSplitter } from 'langchain/text_splitter';
import { OpenAIEmbeddings } from 'langchain/embeddings';
import { PineconeStore } from 'langchain/vectorstores';
import { pinecone } from '@/utils/pinecone-client';
import { PDFLoader } from 'langchain/document_loaders';
import { PINECONE_INDEX_NAME, PINECONE_NAME_SPACE } from '@/config/pinecone';

/* Name of directory to retrieve files from. You can change this as required */
// const filePath = 'docs/02.pdf';
const folder = 'docs'

const getStaticProps = (dirPath: string) => {
  const files = fs.readdirSync(dirPath);
  const pdfs = files.filter((file) => file.endsWith('.pdf'));
  return pdfs;
}



const run = async () => {
  try {
    // iterate through pdfs in folder (client side code)
    // const files = fs.readdirSync(folder);
    /*load raw docs from the pdf file in the directory */
    const pdfs = getStaticProps(folder);
    for (const pdf of pdfs) {
      const filePath = `${folder}/${pdf}`;
      const loader = new PDFLoader(filePath);
      // const loader = new PDFLoader(filePath);
      const rawDocs = await loader.load();

      console.log(rawDocs);

      /* Split text into chunks */
      const textSplitter = new RecursiveCharacterTextSplitter({
        chunkSize: 1000,
        chunkOverlap: 200,
      });

      const docs = await textSplitter.splitDocuments(rawDocs);
      console.log('split docs', docs);

      console.log('creating vector store...');
      /*create and store the embeddings in the vectorStore*/
      const embeddings = new OpenAIEmbeddings();
      const index = pinecone.Index(PINECONE_INDEX_NAME); //change to your own index name
      //embed the PDF documents
      await PineconeStore.fromDocuments(
        index,
        docs,
        embeddings,
        'text',
        PINECONE_NAME_SPACE,
      );
    }
    } catch (error) {
      console.log('error', error);
      throw new Error('Failed to ingest your data');
    }
};

export { run, getStaticProps };

(async () => {
  await run();
  console.log('ingestion complete');
})();
