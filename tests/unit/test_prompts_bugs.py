"""Test cases for prompts.py bugs identified by discriminator."""

import pytest
import sys
sys.path.insert(0, '/Users/akshaypulla/Documents/deal_room_S2P')

from deal_room_S2P.environment.prompts import parse_action_text


class TestDirectMessagePipeTargetExtraction:
    """Bug: direct_message_pipe uses groups[2] for target, but groups[2] is the message."""

    def test_direct_message_pipe_target_is_stakeholder_not_message(self):
        """For 'direct_message Finance | message', target should be 'Finance' not 'message'."""
        text = "direct_message Finance | Here is our pricing details."
        action = parse_action_text(text)
        assert action is not None, f"Failed to parse: {text}"
        assert action.target == "Finance", f"Expected target='Finance', got target='{action.target}'"
        assert action.target_ids == ["Finance"], f"Expected target_ids=['Finance'], got {action.target_ids}"

    def test_direct_message_pipe_message_correct(self):
        """Message for direct_message_pipe should be the actual message, not target."""
        text = "direct_message Legal | We need to address compliance."
        action = parse_action_text(text)
        assert action is not None
        assert action.target == "Legal"
        assert "address compliance" in action.message

    def test_direct_message_no_pipe_still_works(self):
        """direct_message without pipe should still work."""
        text = "direct_message TechLead | Implementation details follow. ###"
        action = parse_action_text(text)
        assert action is not None
        assert action.target == "TechLead"


class TestConcessionPipeTargetExtraction:
    """Bug: concession non-pipe uses groups[1] as term_key (which is stakeholder name)."""

    def test_concession_pipe_with_bar_extracts_target_correctly(self):
        """'concession Finance | price=175000' should have target=Finance."""
        text = "concession Finance | price=175000"
        action = parse_action_text(text)
        assert action is not None, f"Failed to parse: {text}"
        assert action.target == "Finance", f"Expected target='Finance', got target='{action.target}'"
        assert "price" in action.proposed_terms, f"Expected term_key='price', got {list(action.proposed_terms.keys())}"

    def test_concession_pipe_with_message(self):
        """'concession Finance | liability_cap=2000000 We can increase coverage' should parse."""
        text = "concession Finance | liability_cap=2000000 We can increase coverage."
        action = parse_action_text(text)
        assert action is not None
        assert action.target == "Finance"
        assert "liability_cap" in action.proposed_terms
        assert action.proposed_terms["liability_cap"] == 2000000.0

    def test_concession_non_pipe_old_format(self):
        """concession Finance price=175000 (no pipe) should parse term correctly."""
        text = "concession Finance price=175000"
        action = parse_action_text(text)
        assert action is not None, f"Failed to parse: {text}"
        assert action.target == "Finance"
        assert "price" in action.proposed_terms


class TestCleanOutputParsing:
    """Test that prefixed outputs can be cleaned by minimal_grpo_reward._clean_output."""

    def test_clean_output_removes_prefix(self):
        """_clean_output in minimal_grpo_reward should strip prefixes."""
        from deal_room_S2P.environment.minimal_grpo_reward import MinimalDealRoomReward
        cleaner = MinimalDealRoomReward()
        
        texts = [
            "Sure, send_document Legal dpa | Our DPA is attached.",
            "OK direct_message Legal | We are ready to sign.",
            "Certainly, concession Finance | price=175000",
        ]
        for text in texts:
            cleaned = cleaner._clean_output(text)
            print(f"Original: {text[:50]}")
            print(f"Cleaned:  {cleaned[:50]}")
            assert cleaned.startswith(("send_document", "direct_message", "concession")), f"Failed to clean: {text}"


class TestSendDocumentPipe:
    """Test send_document with pipe format."""

    def test_send_document_pipe_target(self):
        """send_document with pipe should extract correct target."""
        text = "send_document Legal dpa | Our DPA covers GDPR obligations."
        action = parse_action_text(text)
        assert action is not None
        assert action.target == "Legal"
        assert action.action_type == "send_document"

    def test_send_document_pipe_doc_type(self):
        """send_document with pipe should extract correct doc_type."""
        text = "send_document Finance roi_model | 3-year NPV analysis."
        action = parse_action_text(text)
        assert action is not None
        assert action.documents[0]["type"] == "roi_model"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
