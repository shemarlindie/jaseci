from jaseci.element.master import master
from jaseci.utils.mem_hook import mem_hook
from jaseci.actor.sentinel import sentinel
from jaseci.graph.graph import graph

from jaseci.utils.utils import TestCaseHelper
from unittest import TestCase


class FasttextClfTests(TestCaseHelper, TestCase):
    """Unit tests for fasttext_clf actions"""

    def setUp(self):
        super().setUp()
        self.master = master(h=mem_hook())
        self.gph = graph(m_id=self.master._m_id, h=self.master._h)
        self.sent = sentinel(m_id=self.master._m_id, h=self.master._h)

    def tearDown(self):
        super().tearDown()

    def test_fasttext_clf_predict(self):
        """test fasttext_clf.predict"""

        jac_code = """
        walker test_fasttext_clf {
            can fasttext_clf.predict;
            has input;

            report fasttext_clf.predict(input);
        }
        """
        self.sent.register_code(jac_code)
        self.assertTrue(self.sent.is_active)

        walker = self.sent.walker_ids.get_obj_by_name('test_fasttext_clf')
        self.assertIsNotNone(walker)

        sentences = ['hello', 'Do I need a passport or visa to enter Guyana?']
        walker.prime(self.gph, {'input': sentences})
        result = walker.run()
        for sentence in sentences:
            self.assertIn(sentence, result[0], sentence)
