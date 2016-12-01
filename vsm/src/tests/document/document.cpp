// Copyright 2016 Yahoo Inc. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.
#include <vespa/vespalib/testkit/testapp.h>

#include <vespa/document/fieldvalue/fieldvalues.h>
#include <vespa/vsm/common/storagedocument.h>
#include <vespa/vespalib/stllike/asciistream.h>


using namespace document;

namespace vsm {

class DocumentTest : public vespalib::TestApp
{
private:
    void testStorageDocument();
    void testStringFieldIdTMap();
public:
    int Main();
};

void
DocumentTest::testStorageDocument()
{
    DocumentType dt("testdoc", 0);

    Field fa("a", 0, *DataType::STRING, true);
    Field fb("b", 1, *DataType::STRING, true);
    dt.addField(fa);
    dt.addField(fb);

    document::Document::UP doc(new document::Document(dt, DocumentId()));
    doc->setValue(fa, StringFieldValue("foo"));
    doc->setValue(fb, StringFieldValue("bar"));

    SharedFieldPathMap fpmap(new FieldPathMapT());
    fpmap->push_back(*dt.buildFieldPath("a"));
    fpmap->push_back(*dt.buildFieldPath("b"));
    fpmap->push_back(FieldPath());
    ASSERT_TRUE((*fpmap)[0].size() == 1);
    ASSERT_TRUE((*fpmap)[1].size() == 1);
    ASSERT_TRUE((*fpmap)[2].size() == 0);

    StorageDocument sdoc(std::move(doc));
    ASSERT_TRUE(sdoc.valid());
    sdoc.setFieldCount(3);
    sdoc.fieldPathMap(fpmap);
    sdoc.init();

    EXPECT_EQUAL(std::string("foo"), sdoc.getField(0)->getAsString());
    EXPECT_EQUAL(std::string("bar"), sdoc.getField(1)->getAsString());
    EXPECT_TRUE(sdoc.getField(2) == NULL);
    // test caching
    EXPECT_EQUAL(std::string("foo"), sdoc.getField(0)->getAsString());
    EXPECT_EQUAL(std::string("bar"), sdoc.getField(1)->getAsString());
    EXPECT_TRUE(sdoc.getField(2) == NULL);

    // set new values
    EXPECT_TRUE(sdoc.setField(0, FieldValue::UP(new StringFieldValue("baz"))));
    EXPECT_EQUAL(std::string("baz"), sdoc.getField(0)->getAsString());
    EXPECT_EQUAL(std::string("bar"), sdoc.getField(1)->getAsString());
    EXPECT_TRUE(sdoc.getField(2) == NULL);
    EXPECT_TRUE(sdoc.setField(1, FieldValue::UP(new StringFieldValue("qux"))));
    EXPECT_EQUAL(std::string("baz"), sdoc.getField(0)->getAsString());
    EXPECT_EQUAL(std::string("qux"), sdoc.getField(1)->getAsString());
    EXPECT_TRUE(sdoc.getField(2) == NULL);
    EXPECT_TRUE(sdoc.setField(2, FieldValue::UP(new StringFieldValue("quux"))));
    EXPECT_EQUAL(std::string("baz"), sdoc.getField(0)->getAsString());
    EXPECT_EQUAL(std::string("qux"), sdoc.getField(1)->getAsString());
    EXPECT_EQUAL(std::string("quux"), sdoc.getField(2)->getAsString());

    // reset cached field values
    sdoc.init();
    EXPECT_EQUAL(std::string("foo"), sdoc.getField(0)->getAsString());
    EXPECT_EQUAL(std::string("bar"), sdoc.getField(1)->getAsString());
    EXPECT_TRUE(sdoc.getField(2) == NULL);

    EXPECT_TRUE(!sdoc.setField(3, FieldValue::UP(new StringFieldValue("thud"))));

    SharedFieldPathMap fim;
    StorageDocument s2(fim);
    EXPECT_EQUAL(vespalib::string("null::"), s2.docDoc().getId().toString());
}

void DocumentTest::testStringFieldIdTMap()
{
    StringFieldIdTMap m;
    EXPECT_EQUAL(0u, m.highestFieldNo());
    EXPECT_TRUE(StringFieldIdTMap::npos == m.fieldNo("unknown"));
    m.add("f1");
    EXPECT_EQUAL(0u, m.fieldNo("f1"));
    EXPECT_EQUAL(1u, m.highestFieldNo());
    m.add("f1");
    EXPECT_EQUAL(0u, m.fieldNo("f1"));
    EXPECT_EQUAL(1u, m.highestFieldNo());
    m.add("f2");
    EXPECT_EQUAL(1u, m.fieldNo("f2"));
    EXPECT_EQUAL(2u, m.highestFieldNo());
    m.add("f3", 7);
    EXPECT_EQUAL(7u, m.fieldNo("f3"));
    EXPECT_EQUAL(8u, m.highestFieldNo());
    m.add("f3");
    EXPECT_EQUAL(7u, m.fieldNo("f3"));
    EXPECT_EQUAL(8u, m.highestFieldNo());
    m.add("f2", 13);
    EXPECT_EQUAL(13u, m.fieldNo("f2"));
    EXPECT_EQUAL(14u, m.highestFieldNo());
    m.add("f4");
    EXPECT_EQUAL(3u, m.fieldNo("f4"));
    EXPECT_EQUAL(14u, m.highestFieldNo());
    {
        vespalib::asciistream os;
        StringFieldIdTMap t;
        t.add("b");
        t.add("a");
        os << t;
        EXPECT_EQUAL(vespalib::string("a = 1\nb = 0\n"), os.str());
    }
    
}

int
DocumentTest::Main()
{
    TEST_INIT("document_test");

    testStorageDocument();
    testStringFieldIdTMap();

    TEST_DONE();
}

}

TEST_APPHOOK(vsm::DocumentTest);

