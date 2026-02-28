import unittest
from unittest.mock import patch, MagicMock
from pageindex import PageIndexClient
import asyncio

class TestPageIndexClient(unittest.TestCase):
    @patch('pageindex.client.md_to_tree')
    def test_index_md(self, mock_md_to_tree):
        mock_md_to_tree.return_value = {
            'doc_name': 'test',
            'doc_description': 'Test description',
            'structure': [{'node_id': '0001', 'title': 'Test Section', 'text': 'Test text'}]
        }
        
        client = PageIndexClient(api_key="dummy")
        
        # Test file needs to exist
        with open("dummy.md", "w") as f:
            f.write("# dummy")
            
        doc_id = client.index("dummy.md")
        self.assertIn(doc_id, client.documents)
        self.assertEqual(client.documents[doc_id]['type'], 'md')
        
    @patch('pageindex.client.tree_retrieve')
    def test_retrieve(self, mock_tree_retrieve):
        mock_tree_retrieve.return_value = [{'node_id': '0001', 'title': 'Test Section', 'text': 'Test text'}]
        
        client = PageIndexClient(api_key="dummy")
        doc_id = "fake_id"
        client.documents[doc_id] = {
            'id': doc_id,
            'path': 'dummy.md',
            'type': 'md',
            'structure': [{'node_id': '0001', 'title': 'Test Section', 'text': 'Test text'}],
        }
        
        results = client.retrieve(doc_id, "What is test?")
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]['node_id'], '0001')
        
    @patch('pageindex.client.ChatGPT_API')
    @patch.object(PageIndexClient, 'retrieve')
    def test_query(self, mock_retrieve, mock_chatgpt):
        mock_retrieve.return_value = [{'node_id': '0001', 'title': 'Test Section', 'text': 'Test text'}]
        mock_chatgpt.return_value = "This is a dummy answer."
        
        client = PageIndexClient(api_key="dummy")
        doc_id = "fake_id"
        
        answer = client.query(doc_id, "What is test?")
        self.assertEqual(answer, "This is a dummy answer.")
        mock_chatgpt.assert_called_once()
        mock_retrieve.assert_called_once_with(doc_id, "What is test?")

if __name__ == '__main__':
    unittest.main()
